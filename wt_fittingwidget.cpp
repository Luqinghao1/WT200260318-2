/*
 * 文件名: wt_fittingwidget.cpp
 * 文件作用: 试井拟合分析主界面实现文件
 * 修改记录:
 * 1. [修复] 修正 FittingDataSettings 成员缺失导致的编译错误。
 * 2. [修复] 修正 loadFittingState 中枚举类型名称错误 (FittingTestType -> WellTestType)。
 * 3. [优化] 加载数据时自动填充物理参数到 Settings，确保保存时数据完整。
 * 4. [修改] 移除了重置参数和更新上下限的按钮槽函数，相关功能移动至参数配置弹窗。
 * 5. [新增] 增加了拟合时间范围的自定义支持 (m_userDefinedTimeMax)。
 * 6. [新增] 支持 Model 19-36 的模型选择与切换逻辑。
 */

#include "wt_fittingwidget.h"
#include "ui_wt_fittingwidget.h"
#include "modelparameter.h"
#include "modelselect.h"
#include "fittingdatadialog.h"
#include "pressurederivativecalculator.h"
#include "pressurederivativecalculator1.h"
#include "paramselectdialog.h"
#include "fittingreport.h"
#include "fittingchart.h"
#include "modelsolver01-06.h"

#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QBuffer>
#include <QFileInfo>
#include <QDateTime>

// 构造函数：初始化界面及相关变量
FittingWidget::FittingWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingWidget),
    m_modelManager(nullptr),
    m_core(new FittingCore(this)),
    m_chartManager(new FittingChart(this)),
    m_mdiArea(nullptr),
    m_chartLogLog(nullptr), m_chartSemiLog(nullptr), m_chartCartesian(nullptr),
    m_subWinLogLog(nullptr), m_subWinSemiLog(nullptr), m_subWinCartesian(nullptr),
    m_plotLogLog(nullptr), m_plotSemiLog(nullptr), m_plotCartesian(nullptr),
    m_currentModelType(ModelManager::Model_1),
    m_obsTime(),
    m_obsDeltaP(),
    m_obsDerivative(),
    m_obsRawP(),
    m_isFitting(false),
    m_isCustomSamplingEnabled(false),
    m_userDefinedTimeMax(-1.0)
{
    ui->setupUi(this);

    // 1. 清理旧布局：如果 plotContainer 中已有布局，先清除，防止重复添加
    if (ui->plotContainer->layout()) {
        QLayoutItem* item;
        while ((item = ui->plotContainer->layout()->takeAt(0)) != nullptr) {
            delete item->widget(); delete item;
        }
        delete ui->plotContainer->layout();
    }
    // 创建新的垂直布局用于放置 MDI 区域
    QVBoxLayout* containerLayout = new QVBoxLayout(ui->plotContainer);
    containerLayout->setContentsMargins(0,0,0,0);
    containerLayout->setSpacing(0);

    // 2. 初始化 MDI 区域 (多文档界面区域)
    m_mdiArea = new QMdiArea(this);
    m_mdiArea->setViewMode(QMdiArea::SubWindowView); // 设置为子窗口模式
    m_mdiArea->setBackground(QBrush(QColor(240, 240, 240))); // 设置背景色
    containerLayout->addWidget(m_mdiArea);

    // 3. 创建三个主要的图表窗口对象
    m_chartLogLog = new FittingChart1(this);     // 双对数图
    m_chartSemiLog = new FittingChart2(this);    // 半对数图
    m_chartCartesian = new FittingChart3(this);  // 历史拟合图

    // 获取底层的绘图控件指针 (QCustomPlot)
    m_plotLogLog = m_chartLogLog->getPlot();
    m_plotSemiLog = m_chartSemiLog->getPlot();
    m_plotCartesian = m_chartCartesian->getPlot();

    // 设置图表标题
    m_chartLogLog->setTitle("双对数曲线 (Log-Log)");
    m_chartSemiLog->setTitle("半对数曲线 (Semi-Log)");
    m_chartCartesian->setTitle("历史拟合曲线 (History Plot)");

    // 将图表窗口添加到 MDI 区域中
    m_subWinLogLog = m_mdiArea->addSubWindow(m_chartLogLog);
    m_subWinSemiLog = m_mdiArea->addSubWindow(m_chartSemiLog);
    m_subWinCartesian = m_mdiArea->addSubWindow(m_chartCartesian);

    // 设置子窗口标题
    m_subWinLogLog->setWindowTitle("双对数图");
    m_subWinSemiLog->setWindowTitle("半对数图");
    m_subWinCartesian->setWindowTitle("标准坐标系");

    // 连接导出数据的信号槽
    connect(m_chartLogLog, &FittingChart1::exportDataTriggered, this, &FittingWidget::onExportCurveData);
    connect(m_chartSemiLog, &FittingChart2::exportDataTriggered, this, &FittingWidget::onExportCurveData);
    connect(m_chartCartesian, &FittingChart3::exportDataTriggered, this, &FittingWidget::onExportCurveData);

    // 连接 m_chartManager 的信号，用于响应手动拟合操作 (例如拖动半对数直线)
    connect(m_chartManager, &FittingChart::sigManualPressureUpdated, this, &FittingWidget::onSemiLogLineMoved);

    // 设置分割器初始比例 (左侧控制面板 : 右侧绘图区)
    ui->splitter->setSizes(QList<int>() << 350 << 650);
    ui->splitter->setCollapsible(0, false); // 禁止完全折叠左侧

    // 4. 初始化参数表格管理器
    m_paramChart = new FittingParameterChart(ui->tableParams, this);

    // 连接参数滚轮修改信号，实现滚动参数时实时刷新曲线
    connect(m_paramChart, &FittingParameterChart::parameterChangedByWheel, this, [this](){
        updateModelCurve(nullptr, false, false);
    });

    // 初始化绘图交互模式
    setupPlot();
    m_chartManager->initializeCharts(m_plotLogLog, m_plotSemiLog, m_plotCartesian);

    // 注册元数据类型，以便在信号槽中传递复杂类型
    qRegisterMetaType<QMap<QString,double>>("QMap<QString,double>");
    qRegisterMetaType<ModelManager::ModelType>("ModelManager::ModelType");
    qRegisterMetaType<QVector<double>>("QVector<double>");

    // 5. 连接拟合核心模块 (FittingCore) 的信号
    connect(m_core, &FittingCore::sigIterationUpdated, this, &FittingWidget::onIterationUpdate, Qt::QueuedConnection);
    connect(m_core, &FittingCore::sigProgress, ui->progressBar, &QProgressBar::setValue);
    connect(m_core, &FittingCore::sigFitFinished, this, &FittingWidget::onFitFinished);

    // 连接界面控件信号
    connect(ui->sliderWeight, &QSlider::valueChanged, this, &FittingWidget::onSliderWeightChanged);
    connect(ui->btnSamplingSettings, &QPushButton::clicked, this, &FittingWidget::onOpenSamplingSettings);

    // 初始化权重滑块
    ui->sliderWeight->setRange(0, 100);
    ui->sliderWeight->setValue(50);
    onSliderWeightChanged(50);
}

// 析构函数：释放 UI 资源
FittingWidget::~FittingWidget()
{
    delete ui;
}

// 槽函数：响应半对数直线移动
// 功能：当用户在半对数图上拖动直线时，更新对应的地层压力 Pi 或 p* 参数
void FittingWidget::onSemiLogLineMoved(double k, double b)
{
    Q_UNUSED(k); // 这里暂时只用到了截距 b
    // 更新参数表中的 Pi (初始地层压力)
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
        m_paramChart->setParameters(params);
    }
}

// 事件处理：显示事件
// 功能：窗口显示时重新布局图表窗口
void FittingWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    layoutCharts();
}

// 设置模型管理器
void FittingWidget::setModelManager(ModelManager *m)
{
    m_modelManager = m;
    m_paramChart->setModelManager(m);
    if (m_core) m_core->setModelManager(m);
    initializeDefaultModel();
}

// 设置项目数据模型映射
void FittingWidget::setProjectDataModels(const QMap<QString, QStandardItemModel *> &models)
{
    m_dataMap = models;
}

// 设置观测数据 (简单版)
void FittingWidget::setObservedData(const QVector<double>& t, const QVector<double>& deltaP, const QVector<double>& d)
{
    setObservedData(t, deltaP, d, QVector<double>());
}

// 设置观测数据 (完整版)
// 功能：存储观测数据，传递给核心模块和图表管理器，并更新曲线显示
void FittingWidget::setObservedData(const QVector<double>& t, const QVector<double>& deltaP,
                                    const QVector<double>& d, const QVector<double>& rawP)
{
    m_obsTime = t;
    m_obsDeltaP = deltaP;
    m_obsDerivative = d;
    m_obsRawP = rawP;

    if (m_core) m_core->setObservedData(t, deltaP, d);
    if (m_chartManager) m_chartManager->setObservedData(t, deltaP, d, rawP);

    updateModelCurve(nullptr, true);
}

// 占位函数：更新基本参数
void FittingWidget::updateBasicParameters() {}

// 初始化默认模型
void FittingWidget::initializeDefaultModel()
{
    if(!m_modelManager) return;
    m_currentModelType = ModelManager::Model_1;

    // [修改] 只显示模型名称，去除 "当前: " 前缀
    ui->btn_modelSelect->setText(ModelManager::getModelTypeName(m_currentModelType));

    // 重置参数、加载项目级参数、隐藏不需要的参数并刷新曲线
    m_paramChart->resetParams(m_currentModelType, true);
    loadProjectParams();
    hideUnwantedParams();
    updateModelCurve(nullptr, true, true);
}

// 设置图表交互模式
void FittingWidget::setupPlot() {
    if(m_plotLogLog) m_plotLogLog->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    if(m_plotSemiLog) m_plotSemiLog->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    if(m_plotCartesian) m_plotCartesian->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
}

// 事件处理：大小改变事件
void FittingWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    layoutCharts();
}

// 重新布局图表窗口
// 功能：将三个图表窗口按固定比例摆放 (左半边放双对数图，右半边上下分放历史图和半对数图)
void FittingWidget::layoutCharts()
{
    if (!m_mdiArea || !m_subWinLogLog || !m_subWinSemiLog || !m_subWinCartesian) return;
    QRect rect = m_mdiArea->contentsRect();
    int w = rect.width();
    int h = rect.height();
    if (w <= 0 || h <= 0) return;

    m_subWinLogLog->setGeometry(0, 0, w / 2, h);
    m_subWinCartesian->setGeometry(w / 2, 0, w - (w / 2), h / 2);
    m_subWinSemiLog->setGeometry(w / 2, h / 2, w - (w / 2), h - (h / 2));

    if (m_subWinLogLog->isMinimized()) m_subWinLogLog->showNormal();
    if (m_subWinCartesian->isMinimized()) m_subWinCartesian->showNormal();
    if (m_subWinSemiLog->isMinimized()) m_subWinSemiLog->showNormal();
}

// 槽函数：加载观测数据
// 功能：打开数据加载对话框，提取选择的数据列，进行预处理 (平滑、导数计算)，并显示
void FittingWidget::on_btnLoadData_clicked() {
    FittingDataDialog dlg(m_dataMap, this);
    if (dlg.exec() != QDialog::Accepted) return;

    FittingDataSettings settings = dlg.getSettings();
    QStandardItemModel* sourceModel = dlg.getPreviewModel();

    if (!sourceModel || sourceModel->rowCount() == 0) {
        QMessageBox::warning(this, "警告", "所选数据源为空，无法加载！");
        return;
    }

    QVector<double> rawTime, rawPressureData, finalDeriv;
    int skip = settings.skipRows;
    int rows = sourceModel->rowCount();

    // 遍历数据源提取时间、压力和导数
    for (int i = skip; i < rows; ++i) {
        QStandardItem* itemT = sourceModel->item(i, settings.timeColIndex);
        QStandardItem* itemP = sourceModel->item(i, settings.pressureColIndex);

        if (itemT && itemP) {
            bool okT, okP;
            double t = itemT->text().toDouble(&okT);
            double p = itemP->text().toDouble(&okP);

            if (okT && okP && t > 0) {
                rawTime.append(t);
                rawPressureData.append(p);
                if (settings.derivColIndex >= 0) {
                    QStandardItem* itemD = sourceModel->item(i, settings.derivColIndex);
                    if (itemD) finalDeriv.append(itemD->text().toDouble());
                    else finalDeriv.append(0.0);
                }
            }
        }
    }

    if (rawTime.isEmpty()) {
        QMessageBox::warning(this, "警告", "未能提取到有效数据。");
        return;
    }

    QVector<double> finalDeltaP;
    double p_shutin = rawPressureData.first();

    // 计算压差 deltaP
    for (double p : rawPressureData) {
        double deltaP = 0.0;
        if (settings.testType == Test_Drawdown) {
            deltaP = std::abs(settings.initialPressure - p);
        } else {
            deltaP = std::abs(p - p_shutin);
        }
        finalDeltaP.append(deltaP);
    }

    // 计算或处理导数
    if (settings.derivColIndex == -1) {
        finalDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(rawTime, finalDeltaP, settings.lSpacing);
        if (settings.enableSmoothing) {
            finalDeriv = PressureDerivativeCalculator1::smoothData(finalDeriv, settings.smoothingSpan);
        }
    } else {
        if (settings.enableSmoothing) {
            finalDeriv = PressureDerivativeCalculator1::smoothData(finalDeriv, settings.smoothingSpan);
        }
        if (finalDeriv.size() != rawTime.size()) {
            finalDeriv.resize(rawTime.size());
        }
    }

    // 将项目级物理参数填充到 Settings 中，以便 Chart 保存和恢复
    ModelParameter* mp = ModelParameter::instance();
    if (mp) {
        settings.porosity = mp->getPhi();
        settings.thickness = mp->getH();
        settings.wellRadius = mp->getRw();
        settings.viscosity = mp->getMu();
        settings.ct = mp->getCt();
        settings.fvf = mp->getB();
        settings.rate = mp->getQ();
    }

    m_chartManager->setSettings(settings);
    setObservedData(rawTime, finalDeltaP, finalDeriv, rawPressureData);

    // [新增] 初始化拟合时间范围为实测数据最大时间
    if (!rawTime.isEmpty()) {
        m_userDefinedTimeMax = rawTime.last();
    }

    QMessageBox::information(this, "成功", "观测数据已成功加载。");
}

// 槽函数：调节拟合权重
// 功能：更新界面上的权重显示
void FittingWidget::onSliderWeightChanged(int value)
{
    double wPressure = value / 100.0;
    double wDerivative = 1.0 - wPressure;
    ui->label_ValDerivative->setText(QString("导数权重: %1").arg(wDerivative, 0, 'f', 2));
    ui->label_ValPressure->setText(QString("压差权重: %1").arg(wPressure, 0, 'f', 2));
}

// 槽函数：打开参数配置对话框
// 功能：显示参数列表，允许用户配置拟合参数、上下限以及拟合时间范围
void FittingWidget::on_btnSelectParams_clicked()
{
    // 同步表格中的最新数据到内部列表
    m_paramChart->updateParamsFromTable();
    QList<FitParameter> currentParams = m_paramChart->getParameters();

    // [新增] 确定传入对话框的拟合时间
    // 如果尚未设置或为负，优先取实测数据最大值，若无实测数据则默认 10000
    double currentTime = m_userDefinedTimeMax;
    if (currentTime <= 0 && !m_obsTime.isEmpty()) {
        currentTime = m_obsTime.last();
    } else if (currentTime <= 0) {
        currentTime = 10000.0;
    }

    // 打开参数选择对话框，传入当前参数、模型类型和拟合时间
    ParamSelectDialog dlg(currentParams, m_currentModelType, currentTime, this);

    if(dlg.exec() == QDialog::Accepted) {
        // 更新参数
        QList<FitParameter> updatedParams = dlg.getUpdatedParams();
        for(auto& p : updatedParams) {
            // LfD 为计算参数，强制不拟合
            if(p.name == "LfD") p.isFit = false;
        }
        m_paramChart->setParameters(updatedParams);

        // [新增] 获取并更新用户设定的拟合时间范围
        m_userDefinedTimeMax = dlg.getFittingTime();

        hideUnwantedParams();
        // 刷新曲线
        updateModelCurve(nullptr, false);
    }
}

// 辅助函数：隐藏不需要显示的参数
void FittingWidget::hideUnwantedParams()
{
    for(int i = 0; i < ui->tableParams->rowCount(); ++i) {
        QTableWidgetItem* item = ui->tableParams->item(i, 1);
        if(item) {
            QString name = item->data(Qt::UserRole).toString();
            // 隐藏辅助参数 LfD
            if(name == "LfD") {
                ui->tableParams->setRowHidden(i, true);
            }
        }
    }
}

// 槽函数：打开抽样设置
void FittingWidget::onOpenSamplingSettings()
{
    if (m_obsTime.isEmpty()) {
        QMessageBox::warning(this, "提示", "请先加载观测数据，以便确定时间范围。");
        return;
    }
    double tMin = m_obsTime.first();
    double tMax = m_obsTime.last();

    SamplingSettingsDialog dlg(m_customIntervals, m_isCustomSamplingEnabled, tMin, tMax, this);
    if (dlg.exec() == QDialog::Accepted) {
        m_customIntervals = dlg.getIntervals();
        m_isCustomSamplingEnabled = dlg.isCustomSamplingEnabled();
        if(m_core) m_core->setSamplingSettings(m_customIntervals, m_isCustomSamplingEnabled);
        updateModelCurve(nullptr, false);
    }
}

// 槽函数：开始自动拟合
// 功能：收集参数和配置，调用核心模块开始回归计算
void FittingWidget::on_btnRunFit_clicked() {
    if(m_isFitting) return;
    if(m_obsTime.isEmpty()) {
        QMessageBox::warning(this,"错误","请先加载观测数据。");
        return;
    }
    m_paramChart->updateParamsFromTable();
    m_isFitting = true;
    ui->btnRunFit->setEnabled(false);
    ui->btnSelectParams->setEnabled(false);

    ModelManager::ModelType modelType = m_currentModelType;
    QList<FitParameter> paramsCopy = m_paramChart->getParameters();
    double w = ui->sliderWeight->value() / 100.0;
    if(m_core) m_core->startFit(modelType, paramsCopy, w);
}

// 槽函数：停止拟合
void FittingWidget::on_btnStop_clicked() {
    if(m_core) m_core->stopFit();
}

// 槽函数：理论模型更新按钮
// 功能：强制刷新理论曲线 (和参数滚轮修改类似)
void FittingWidget::on_btnImportModel_clicked() {
    updateModelCurve(nullptr, false, false);
}

// 槽函数：选择模型
// 功能：弹出模型选择对话框，切换当前使用的试井模型
// [修改] 增加对 Model 19-36 的支持
void FittingWidget::on_btn_modelSelect_clicked() {
    ModelSelect dlg(this);

    // 传入当前模型代码以便回显，格式为 "modelwidgetX"
    int currentId = (int)m_currentModelType + 1;
    dlg.setCurrentModelCode(QString("modelwidget%1").arg(currentId));

    if (dlg.exec() == QDialog::Accepted) {
        QString code = dlg.getSelectedModelCode();

        // [修改] 通用解析逻辑：modelwidgetX -> X
        QString numStr = code;
        numStr.remove("modelwidget");
        bool ok;
        int modelId = numStr.toInt(&ok);

        if (ok && modelId >= 1 && modelId <= 36) {
            // 将 ID 转换为内部枚举值 (0-35)
            ModelManager::ModelType newType = (ModelManager::ModelType)(modelId - 1);

            // 切换参数管理器中的模型
            m_paramChart->switchModel(newType);
            m_currentModelType = newType;

            // 更新按钮文本
            ui->btn_modelSelect->setText(ModelManager::getModelTypeName(newType));

            // 加载默认参数及处理显示逻辑
            loadProjectParams();
            hideUnwantedParams();
            updateModelCurve(nullptr, true);
        } else {
            QMessageBox::warning(this, "提示", "所选组合暂无对应的模型。\nCode: " + code);
        }
    }
}

// 辅助函数：加载项目级参数
// 功能：从 ModelParameter 单例中读取孔隙度、厚度等物理参数同步到拟合界面
void FittingWidget::loadProjectParams()
{
    ModelParameter* mp = ModelParameter::instance();
    QList<FitParameter> params = m_paramChart->getParameters();
    bool changed = false;
    for(auto& p : params) {
        if(p.name == "phi") { p.value = mp->getPhi(); changed = true; }
        else if(p.name == "h") { p.value = mp->getH(); changed = true; }
        else if(p.name == "rw") { p.value = mp->getRw(); changed = true; }
        else if(p.name == "mu") { p.value = mp->getMu(); changed = true; }
        else if(p.name == "Ct") { p.value = mp->getCt(); changed = true; }
        else if(p.name == "B") { p.value = mp->getB(); changed = true; }
        else if(p.name == "q") { p.value = mp->getQ(); changed = true; }
    }
    if(changed) m_paramChart->setParameters(params);
}

// 核心函数：更新理论曲线
// 功能：根据当前参数计算理论曲线，并支持敏感性分析模式
void FittingWidget::updateModelCurve(const QMap<QString, double>* explicitParams, bool autoScale, bool calcError) {
    if(!m_modelManager) {
        QMessageBox::critical(this, "错误", "ModelManager 未初始化！");
        return;
    }
    // 如果没有加载数据，也要能画出理论曲线 (空数据模式下)
    if(m_obsTime.isEmpty() && !explicitParams && m_userDefinedTimeMax <= 0) {
        // 如果真的没有任何时间参考，则清空
        m_chartLogLog->clearGraphs();
        m_chartSemiLog->clearGraphs();
        m_chartCartesian->clearGraphs();
        return;
    }

    ui->tableParams->clearFocus();

    // 收集参数
    QMap<QString, double> rawParams;
    QString sensitivityKey = "";
    QVector<double> sensitivityValues;

    if (explicitParams) {
        rawParams = *explicitParams;
    } else {
        QList<FitParameter> allParams = m_paramChart->getParameters();
        for(const auto& p : allParams) rawParams.insert(p.name, p.value);
        QMap<QString, QString> rawTexts = m_paramChart->getRawParamTexts();
        for(auto it = rawTexts.begin(); it != rawTexts.end(); ++it) {
            QVector<double> vals = parseSensitivityValues(it.value());
            if (!vals.isEmpty()) {
                rawParams.insert(it.key(), vals.first());
                if (vals.size() > 1 && sensitivityKey.isEmpty()) {
                    sensitivityKey = it.key();
                    sensitivityValues = vals;
                }
            } else { rawParams.insert(it.key(), 0.0); }
        }
    }

    // 预处理参数 (转换为求解器所需格式)
    QMap<QString, double> solverParams = FittingCore::preprocessParams(rawParams, m_currentModelType);

    // [修改] 生成时间步长逻辑
    QVector<double> targetT;

    // 确定计算的最大时间
    double tMax = 10000.0;
    if (m_userDefinedTimeMax > 0) {
        tMax = m_userDefinedTimeMax; // 优先使用用户设定值
    } else if (!m_obsTime.isEmpty()) {
        tMax = m_obsTime.last();
    }

    // 确定最小时间
    double tMin = (!m_obsTime.isEmpty()) ? std::max(1e-5, m_obsTime.first()) : 1e-4;

    // 如果用户设置的时间小于实测时间起始，修正范围防止错误
    if (tMax < tMin) tMax = tMin * 10.0;

    targetT = ModelManager::generateLogTimeSteps(300, log10(tMin), log10(tMax));

    bool isSensitivityMode = !sensitivityKey.isEmpty();
    ui->btnRunFit->setEnabled(!isSensitivityMode);

    if (isSensitivityMode) {
        // 敏感性分析模式：绘制多条曲线
        ui->label_Error->setText(QString("敏感性分析模式: %1 (%2 个值)").arg(sensitivityKey).arg(sensitivityValues.size()));
        m_chartLogLog->clearGraphs();
        m_chartManager->plotAll(QVector<double>(), QVector<double>(), QVector<double>(), false, autoScale);

        QList<QColor> colors = { Qt::red, Qt::blue, QColor(0,180,0), Qt::magenta, QColor(255,140,0), Qt::cyan, Qt::darkRed, Qt::darkBlue };
        for(int i = 0; i < sensitivityValues.size(); ++i) {
            double val = sensitivityValues[i];
            QMap<QString, double> currentParams = rawParams;
            currentParams[sensitivityKey] = val;
            QMap<QString, double> currentSolverParams = FittingCore::preprocessParams(currentParams, m_currentModelType);

            ModelCurveData res = m_modelManager->calculateTheoreticalCurve(m_currentModelType, currentSolverParams, targetT);
            QColor c = colors[i % colors.size()];
            QString suffix = QString("%1=%2").arg(sensitivityKey).arg(val);
            QCPGraph* gP = m_plotLogLog->addGraph();
            gP->setData(std::get<0>(res), std::get<1>(res));
            gP->setPen(QPen(c, 2)); gP->setName("P: "+suffix);
            QCPGraph* gD = m_plotLogLog->addGraph();
            gD->setData(std::get<0>(res), std::get<2>(res));
            gD->setPen(QPen(c, 2, Qt::DashLine)); gD->setName("P': "+suffix);
        }
    } else {
        // 正常模式：绘制单条曲线
        ModelCurveData res = m_modelManager->calculateTheoreticalCurve(m_currentModelType, solverParams, targetT);
        m_chartManager->plotAll(std::get<0>(res), std::get<1>(res), std::get<2>(res), true, autoScale);

        // 计算误差
        if (!m_obsTime.isEmpty() && m_core && calcError) {
            QVector<double> sampleT, sampleP, sampleD;
            m_core->getLogSampledData(m_obsTime, m_obsDeltaP, m_obsDerivative, sampleT, sampleP, sampleD);
            QVector<double> residuals = m_core->calculateResiduals(rawParams, m_currentModelType, ui->sliderWeight->value()/100.0, sampleT, sampleP, sampleD);
            double sse = m_core->calculateSumSquaredError(residuals);
            ui->label_Error->setText(QString("误差(MSE): %1").arg(sse/residuals.size(), 0, 'e', 3));
            if (m_isCustomSamplingEnabled) m_chartManager->plotSampledPoints(sampleT, sampleP, sampleD);
        }
    }
    m_plotLogLog->replot();
    m_plotSemiLog->replot();
    m_plotCartesian->replot();
}

// 槽函数：迭代更新
// 功能：在自动拟合过程中，接收每次迭代的结果并刷新图表
void FittingWidget::onIterationUpdate(double err, const QMap<QString,double>& p,
                                      const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve) {
    ui->label_Error->setText(QString("误差(MSE): %1").arg(err, 0, 'e', 3));
    ui->tableParams->blockSignals(true);
    // 更新表格中的参数值
    for(int i=0; i<ui->tableParams->rowCount(); ++i) {
        QString key = ui->tableParams->item(i, 1)->data(Qt::UserRole).toString();
        if(p.contains(key)) ui->tableParams->item(i, 2)->setText(QString::number(p[key], 'g', 5));
    }
    ui->tableParams->blockSignals(false);

    // 刷新曲线
    m_chartManager->plotAll(t, p_curve, d_curve, true, false);
    if (m_isCustomSamplingEnabled && m_core) {
        QVector<double> sampleT, sampleP, sampleD;
        m_core->getLogSampledData(m_obsTime, m_obsDeltaP, m_obsDerivative, sampleT, sampleP, sampleD);
        m_chartManager->plotSampledPoints(sampleT, sampleP, sampleD);
    }
    if(m_plotLogLog) m_plotLogLog->replot();
    if(m_plotSemiLog) m_plotSemiLog->replot();
    if(m_plotCartesian) m_plotCartesian->replot();
}

// 槽函数：拟合完成
void FittingWidget::onFitFinished() {
    m_isFitting = false;
    ui->btnRunFit->setEnabled(true);
    ui->btnSelectParams->setEnabled(true);
    QMessageBox::information(this, "完成", "拟合完成。");
}

// 槽函数：导出拟合参数
void FittingWidget::on_btnExportData_clicked() {
    m_paramChart->updateParamsFromTable();
    QList<FitParameter> params = m_paramChart->getParameters();
    QString defaultDir = ModelParameter::instance()->getProjectPath();
    if(defaultDir.isEmpty()) defaultDir = ".";
    QString fileName = QFileDialog::getSaveFileName(this, "导出拟合参数", defaultDir + "/FittingParameters.csv", "CSV Files (*.csv);;Text Files (*.txt)");
    if (fileName.isEmpty()) return;
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream out(&file);
    if(fileName.endsWith(".csv", Qt::CaseInsensitive)) {
        file.write("\xEF\xBB\xBF"); // BOM for UTF-8
        out << QString("参数中文名,参数英文名,拟合值,单位\n");
        for(const auto& param : params) {
            QString htmlSym, uniSym, unitStr, dummyName;
            FittingParameterChart::getParamDisplayInfo(param.name, dummyName, htmlSym, uniSym, unitStr);
            if(unitStr == "无因次" || unitStr == "小数") unitStr = "";
            out << QString("%1,%2,%3,%4\n").arg(param.displayName).arg(uniSym).arg(param.value, 0, 'g', 10).arg(unitStr);
        }
    } else {
        for(const auto& param : params) {
            QString htmlSym, uniSym, unitStr, dummyName;
            FittingParameterChart::getParamDisplayInfo(param.name, dummyName, htmlSym, uniSym, unitStr);
            if(unitStr == "无因次" || unitStr == "小数") unitStr = "";
            out << QString("%1 (%2): %3 %4").arg(param.displayName).arg(uniSym).arg(param.value, 0, 'g', 10).arg(unitStr) << "\n";
        }
    }
    file.close();
    QMessageBox::information(this, "完成", "参数数据已成功导出。");
}

// 槽函数：导出曲线数据
void FittingWidget::onExportCurveData() {
    QString defaultDir = ModelParameter::instance()->getProjectPath();
    if(defaultDir.isEmpty()) defaultDir = ".";
    QString path = QFileDialog::getSaveFileName(this, "导出拟合曲线数据", defaultDir + "/FittingCurves.csv", "CSV Files (*.csv)");
    if (path.isEmpty()) return;

    // 获取图表中的数据指针
    auto graphObsP = m_plotLogLog->graph(0);
    auto graphObsD = m_plotLogLog->graph(1);
    if (!graphObsP) return;
    QCPGraph *graphModP = (m_plotLogLog->graphCount() > 2) ? m_plotLogLog->graph(2) : nullptr;
    QCPGraph *graphModD = (m_plotLogLog->graphCount() > 3) ? m_plotLogLog->graph(3) : nullptr;

    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << "Obs_Time,Obs_DP,Obs_Deriv,Model_Time,Model_DP,Model_Deriv\n";

        auto itObsP = graphObsP->data()->begin();
        auto itObsD = graphObsD->data()->begin();
        auto endObsP = graphObsP->data()->end();

        QCPGraphDataContainer::const_iterator itModP, endModP, itModD;
        bool hasModel = (graphModP != nullptr && graphModD != nullptr);
        if(hasModel) { itModP = graphModP->data()->begin(); endModP = graphModP->data()->end(); itModD = graphModD->data()->begin(); }

        while (itObsP != endObsP || (hasModel && itModP != endModP)) {
            QStringList line;
            if (itObsP != endObsP) {
                line << QString::number(itObsP->key, 'g', 10) << QString::number(itObsP->value, 'g', 10);
                if (itObsD != graphObsD->data()->end()) { line << QString::number(itObsD->value, 'g', 10); ++itObsD; } else { line << ""; }
                ++itObsP;
            } else { line << "" << "" << ""; }

            if (hasModel && itModP != endModP) {
                line << QString::number(itModP->key, 'g', 10) << QString::number(itModP->value, 'g', 10);
                if (itModD != graphModD->data()->end()) { line << QString::number(itModD->value, 'g', 10); ++itModD; } else { line << ""; }
                ++itModP;
            } else { line << "" << "" << ""; }
            out << line.join(",") << "\n";
        }
        f.close();
        QMessageBox::information(this, "导出成功", "拟合曲线数据已保存。");
    }
}

// 槽函数：导出报告
void FittingWidget::on_btnExportReport_clicked() {
    QString wellName = "未命名井";
    QString projectFilePath = ModelParameter::instance()->getProjectFilePath();
    QFile pwtFile(projectFilePath);
    if (pwtFile.exists() && pwtFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QByteArray data = pwtFile.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (!doc.isNull() && doc.isObject()) {
            QJsonObject root = doc.object();
            if (root.contains("wellName")) wellName = root["wellName"].toString();
            else if (root.contains("basicParams")) {
                QJsonObject basic = root["basicParams"].toObject();
                if (basic.contains("wellName")) wellName = basic["wellName"].toString();
            }
        }
        pwtFile.close();
    }
    if (wellName == "未命名井" || wellName.isEmpty()) { QFileInfo fi(projectFilePath); wellName = fi.completeBaseName(); }

    FittingReportData reportData;
    reportData.wellName = wellName;
    reportData.modelType = m_currentModelType;
    QString mseText = ui->label_Error->text().remove("误差(MSE): ");
    reportData.mse = mseText.toDouble();
    reportData.t = m_obsTime; reportData.p = m_obsDeltaP; reportData.d = m_obsDerivative;
    m_paramChart->updateParamsFromTable();
    reportData.params = m_paramChart->getParameters();
    reportData.imgLogLog = getPlotImageBase64(m_plotLogLog);
    reportData.imgSemiLog = getPlotImageBase64(m_plotSemiLog);
    reportData.imgCartesian = getPlotImageBase64(m_plotCartesian);

    QString reportFileName = QString("%1试井解释报告.doc").arg(wellName);
    QString defaultDir = QFileInfo(projectFilePath).absolutePath();
    if(defaultDir.isEmpty() || defaultDir == ".") defaultDir = ModelParameter::instance()->getProjectPath();
    if(defaultDir.isEmpty()) defaultDir = ".";

    QString fileName = QFileDialog::getSaveFileName(this, "导出报告", defaultDir + "/" + reportFileName, "Word 文档 (*.doc);;HTML 文件 (*.html)");
    if(fileName.isEmpty()) return;

    QString errorMsg;
    if (FittingReportGenerator::generate(fileName, reportData, &errorMsg)) {
        QMessageBox::information(this, "成功", QString("报告及数据已导出！\n\n文件路径: %1").arg(fileName));
    } else {
        QMessageBox::critical(this, "错误", "报告导出失败:\n" + errorMsg);
    }
}

// 辅助函数：将图表转换为 Base64 图像
QString FittingWidget::getPlotImageBase64(MouseZoom* plot) {
    if(!plot) return "";
    QPixmap pixmap = plot->toPixmap(800, 600);
    QByteArray byteArray;
    QBuffer buffer(&byteArray);
    buffer.open(QIODevice::WriteOnly);
    pixmap.save(&buffer, "PNG");
    return QString::fromLatin1(byteArray.toBase64().data());
}

// 槽函数：保存拟合状态
void FittingWidget::on_btnSaveFit_clicked()
{
    emit sigRequestSave();
}

// 获取拟合状态的 JSON 对象
QJsonObject FittingWidget::getJsonState() const
{
    const_cast<FittingWidget*>(this)->m_paramChart->updateParamsFromTable();
    QList<FitParameter> params = m_paramChart->getParameters();

    QJsonObject root;
    root["modelType"] = (int)m_currentModelType;
    root["modelName"] = ModelManager::getModelTypeName(m_currentModelType);
    root["fitWeightVal"] = ui->sliderWeight->value();

    QJsonObject plotRange;
    plotRange["xMin"] = m_plotLogLog->xAxis->range().lower;
    plotRange["xMax"] = m_plotLogLog->xAxis->range().upper;
    plotRange["yMin"] = m_plotLogLog->yAxis->range().lower;
    plotRange["yMax"] = m_plotLogLog->yAxis->range().upper;
    root["plotView"] = plotRange;

    // 保存半对数图视野
    QJsonObject semiLogRange;
    semiLogRange["xMin"] = m_plotSemiLog->xAxis->range().lower;
    semiLogRange["xMax"] = m_plotSemiLog->xAxis->range().upper;
    semiLogRange["yMin"] = m_plotSemiLog->yAxis->range().lower;
    semiLogRange["yMax"] = m_plotSemiLog->yAxis->range().upper;
    root["plotViewSemiLog"] = semiLogRange;

    // 保存数据配置
    if (m_chartManager) {
        FittingDataSettings settings = m_chartManager->getSettings();
        QJsonObject settingsObj;
        settingsObj["producingTime"] = settings.producingTime;
        settingsObj["initialPressure"] = settings.initialPressure;
        settingsObj["testType"] = (int)settings.testType;
        settingsObj["porosity"] = settings.porosity;
        settingsObj["thickness"] = settings.thickness;
        settingsObj["wellRadius"] = settings.wellRadius;
        settingsObj["viscosity"] = settings.viscosity;
        settingsObj["ct"] = settings.ct;
        settingsObj["fvf"] = settings.fvf;
        settingsObj["rate"] = settings.rate;
        settingsObj["skipRows"] = settings.skipRows;
        settingsObj["timeCol"] = settings.timeColIndex;
        settingsObj["presCol"] = settings.pressureColIndex;
        settingsObj["derivCol"] = settings.derivColIndex;
        settingsObj["lSpacing"] = settings.lSpacing;
        settingsObj["smoothing"] = settings.enableSmoothing;
        settingsObj["span"] = settings.smoothingSpan;
        root["dataSettings"] = settingsObj;
    }

    QJsonArray paramsArray;
    for(const auto& p : params) {
        QJsonObject pObj;
        pObj["name"] = p.name;
        pObj["value"] = p.value;
        pObj["isFit"] = p.isFit;
        pObj["min"] = p.min;
        pObj["max"] = p.max;
        pObj["isVisible"] = p.isVisible;
        pObj["step"] = p.step;
        paramsArray.append(pObj);
    }
    root["parameters"] = paramsArray;

    QJsonArray timeArr, pressArr, derivArr, rawPArr;
    for(double v : m_obsTime) timeArr.append(v);
    for(double v : m_obsDeltaP) pressArr.append(v);
    for(double v : m_obsDerivative) derivArr.append(v);
    for(double v : m_obsRawP) rawPArr.append(v);

    QJsonObject obsData;
    obsData["time"] = timeArr;
    obsData["pressure"] = pressArr;
    obsData["derivative"] = derivArr;
    obsData["rawPressure"] = rawPArr;
    root["observedData"] = obsData;

    root["useCustomSampling"] = m_isCustomSamplingEnabled;
    QJsonArray intervalArr;
    for(const auto& item : m_customIntervals) {
        QJsonObject obj;
        obj["start"] = item.tStart;
        obj["end"] = item.tEnd;
        obj["count"] = item.count;
        intervalArr.append(obj);
    }
    root["customIntervals"] = intervalArr;

    // 保存手动拟合结果
    if (m_chartManager) {
        root["manualPressureFitState"] = m_chartManager->getManualPressureState();
    }

    // [新增] 保存用户自定义的拟合时间范围
    root["fittingTimeMax"] = m_userDefinedTimeMax;

    return root;
}

// 加载拟合状态
void FittingWidget::loadFittingState(const QJsonObject& root)
{
    if (root.isEmpty()) return;

    if (root.contains("modelType")) {
        int type = root["modelType"].toInt();
        m_currentModelType = (ModelManager::ModelType)type;

        // [修改] 只显示模型名称，去除 "当前: "
        ui->btn_modelSelect->setText(ModelManager::getModelTypeName(m_currentModelType));
    }

    m_paramChart->resetParams(m_currentModelType);

    // 优先加载数据配置
    if (root.contains("dataSettings") && m_chartManager) {
        QJsonObject sObj = root["dataSettings"].toObject();
        FittingDataSettings settings;
        settings.producingTime = sObj["producingTime"].toDouble();
        settings.initialPressure = sObj["initialPressure"].toDouble();
        settings.testType = (WellTestType)sObj["testType"].toInt();
        settings.porosity = sObj["porosity"].toDouble();
        settings.thickness = sObj["thickness"].toDouble();
        settings.wellRadius = sObj["wellRadius"].toDouble();
        settings.viscosity = sObj["viscosity"].toDouble();
        settings.ct = sObj["ct"].toDouble();
        settings.fvf = sObj["fvf"].toDouble();
        settings.rate = sObj["rate"].toDouble();
        settings.skipRows = sObj["skipRows"].toInt();
        settings.timeColIndex = sObj["timeCol"].toInt();
        settings.pressureColIndex = sObj["presCol"].toInt();
        settings.derivColIndex = sObj["derivCol"].toInt();
        settings.lSpacing = sObj["lSpacing"].toDouble();
        settings.enableSmoothing = sObj["smoothing"].toBool();
        settings.smoothingSpan = sObj["span"].toDouble();

        m_chartManager->setSettings(settings);
    }

    QMap<QString, double> explicitParamsMap;
    if (root.contains("parameters")) {
        QJsonArray arr = root["parameters"].toArray();
        QList<FitParameter> currentParams = m_paramChart->getParameters();
        for(int i=0; i<arr.size(); ++i) {
            QJsonObject pObj = arr[i].toObject();
            QString name = pObj["name"].toString();
            for(auto& p : currentParams) {
                if(p.name == name) {
                    p.value = pObj["value"].toDouble();
                    p.isFit = pObj["isFit"].toBool();
                    p.min = pObj["min"].toDouble();
                    p.max = pObj["max"].toDouble();
                    if(pObj.contains("isVisible")) p.isVisible = pObj["isVisible"].toBool();
                    else p.isVisible = true;
                    if(pObj.contains("step")) p.step = pObj["step"].toDouble();
                    explicitParamsMap.insert(p.name, p.value);
                    break;
                }
            }
        }
        m_paramChart->setParameters(currentParams);
    }

    if (root.contains("fitWeightVal")) ui->sliderWeight->setValue(root["fitWeightVal"].toInt());

    if (root.contains("observedData")) {
        QJsonObject obs = root["observedData"].toObject();
        QJsonArray tArr = obs["time"].toArray();
        QJsonArray pArr = obs["pressure"].toArray();
        QJsonArray dArr = obs["derivative"].toArray();
        QJsonArray rawPArr;
        if (obs.contains("rawPressure")) rawPArr = obs["rawPressure"].toArray();

        QVector<double> t, p, d, rawP;
        for(auto v : tArr) t.append(v.toDouble());
        for(auto v : pArr) p.append(v.toDouble());
        for(auto v : dArr) d.append(v.toDouble());
        for(auto v : rawPArr) rawP.append(v.toDouble());

        setObservedData(t, p, d, rawP);
    }

    if (root.contains("useCustomSampling")) m_isCustomSamplingEnabled = root["useCustomSampling"].toBool();
    if (root.contains("customIntervals")) {
        m_customIntervals.clear();
        QJsonArray arr = root["customIntervals"].toArray();
        for(auto v : arr) {
            QJsonObject obj = v.toObject();
            SamplingInterval item;
            item.tStart = obj["start"].toDouble();
            item.tEnd = obj["end"].toDouble();
            item.count = obj["count"].toInt();
            m_customIntervals.append(item);
        }
        if(m_core) m_core->setSamplingSettings(m_customIntervals, m_isCustomSamplingEnabled);
    }

    // [新增] 加载用户自定义的拟合时间范围
    if (root.contains("fittingTimeMax")) {
        m_userDefinedTimeMax = root["fittingTimeMax"].toDouble();
    } else {
        m_userDefinedTimeMax = -1.0;
    }

    hideUnwantedParams();

    updateModelCurve(&explicitParamsMap);

    if (root.contains("plotView")) {
        QJsonObject range = root["plotView"].toObject();
        if (range.contains("xMin") && range.contains("xMax")) {
            double xMin = range["xMin"].toDouble();
            double xMax = range["xMax"].toDouble();
            double yMin = range["yMin"].toDouble();
            double yMax = range["yMax"].toDouble();
            if (xMax > xMin && yMax > yMin && xMin > 0 && yMin > 0) {
                m_plotLogLog->xAxis->setRange(xMin, xMax);
                m_plotLogLog->yAxis->setRange(yMin, yMax);
                m_plotLogLog->replot();
            }
        }
    }

    if (root.contains("plotViewSemiLog")) {
        QJsonObject range = root["plotViewSemiLog"].toObject();
        if (range.contains("xMin") && range.contains("xMax")) {
            double xMin = range["xMin"].toDouble();
            double xMax = range["xMax"].toDouble();
            double yMin = range["yMin"].toDouble();
            double yMax = range["yMax"].toDouble();
            if (xMax > xMin || (xMax < xMin)) {
                m_plotSemiLog->xAxis->setRange(xMin, xMax);
                m_plotSemiLog->yAxis->setRange(yMin, yMax);
                m_plotSemiLog->replot();
            }
        }
    }

    if (root.contains("manualPressureFitState") && m_chartManager) {
        m_chartManager->setManualPressureState(root["manualPressureFitState"].toObject());
    }
}

// 辅助函数：解析敏感性分析输入
QVector<double> FittingWidget::parseSensitivityValues(const QString& text) {
    QVector<double> values;
    QString cleanText = text;
    cleanText.replace(QChar(0xFF0C), ","); // 替换中文逗号
    QStringList parts = cleanText.split(',', Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok;
        double v = part.trimmed().toDouble(&ok);
        if (ok) values.append(v);
    }
    return values;
}
