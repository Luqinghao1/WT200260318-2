/*
 * 文件名: mainwindow.cpp
 * 文件作用: 主窗口类实现文件
 * 功能描述:
 * 1. 负责应用程序的整体初始化和页面布局，建立各个子界面的连接。
 * 2. 实现了左侧导航栏的逻辑控制和页面切换。
 * 3. 协调数据在不同模块之间的流转，确保项目数据能顺利同步到下游各个分析模块。
 * 4. 初始化多数据拟合标签页管理器（multidatafittingpage）以替换原有的预测界面，实现多数据功能入口。
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "navbtn.h"
#include "wt_projectwidget.h"
#include "wt_datawidget.h"
#include "modelmanager.h"
#include "modelparameter.h"
#include "wt_plottingwidget.h"
#include "fittingpage.h"
#include "settingswidget.h"
#include "pressurederivativecalculator.h"

// 【修改点1】引入改名后的多数据拟合标签页管理器头文件
#include "multidatafittingpage.h"

#include <QDateTime>
#include <QMessageBox>
#include <QDebug>
#include <QStandardItemModel>
#include <QTimer>
#include <QSpacerItem>
#include <QStackedWidget>
#include <cmath>
#include <QStatusBar>

// 辅助函数：统一的消息框样式定义
static QString getGlobalMessageBoxStyle()
{
    return "QMessageBox { background-color: #ffffff; color: #000000; }"
           "QLabel { color: #000000; background-color: transparent; }"
           "QPushButton { "
           "   color: #000000; "
           "   background-color: #f0f0f0; "
           "   border: 1px solid #c0c0c0; "
           "   border-radius: 3px; "
           "   padding: 5px 15px; "
           "   min-width: 60px;"
           "}"
           "QPushButton:hover { background-color: #e0e0e0; }"
           "QPushButton:pressed { background-color: #d0d0d0; }";
}

// 构造函数：初始化主窗口UI，设置标题和最小宽度，并调用初始化流程
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_isProjectLoaded(false)
{
    ui->setupUi(this);
    this->setWindowTitle("PWT页岩油多级压裂水平井试井解释软件");
    this->setMinimumWidth(1024);

    init();
}

// 析构函数：释放UI资源
MainWindow::~MainWindow()
{
    delete ui;
}

// 核心初始化函数：初始化左侧导航栏、状态栏时间以及所有子页面模块
void MainWindow::init()
{
    // --- 初始化左侧导航栏 ---
    for(int i = 0 ; i < 7; i++)
    {
        NavBtn* btn = new NavBtn(ui->widgetNav);
        btn->setMinimumWidth(110);
        btn->setIndex(i);
        btn->setStyleSheet("color: black;");

        // 配置导航按钮图标及对应页面索引
        switch (i) {
        case 0:
            btn->setPicName("border-image: url(:/new/prefix1/Resource/X0.png);", tr("项目"));
            btn->setClickedStyle();
            ui->stackedWidget->setCurrentIndex(0);
            break;
        case 1: btn->setPicName("border-image: url(:/new/prefix1/Resource/X1.png);", tr("数据")); break;
        case 2: btn->setPicName("border-image: url(:/new/prefix1/Resource/X2.png);", tr("图表")); break;
        case 3: btn->setPicName("border-image: url(:/new/prefix1/Resource/X3.png);", tr("模型")); break;
        case 4: btn->setPicName("border-image: url(:/new/prefix1/Resource/X4.png);", tr("拟合")); break;
        // 将原“预测”按钮修改为“多数据”按钮
        case 5: btn->setPicName("border-image: url(:/new/prefix1/Resource/X5.png);", tr("多数据")); break;
        case 6: btn->setPicName("border-image: url(:/new/prefix1/Resource/X6.png);", tr("设置")); break;
        }
        m_NavBtnMap.insert(btn->getName(), btn);
        ui->verticalLayoutNav->addWidget(btn);

        // 导航按钮点击事件处理
        connect(btn, &NavBtn::sigClicked, [=](QString name)
                {
                    int targetIndex = m_NavBtnMap.value(name)->getIndex();

                    // 权限检查：必须有活动项目才能进入数据处理相关页面
                    if ((targetIndex >= 1 && targetIndex <= 5) && !m_isProjectLoaded) {
                        QMessageBox msgBox;
                        msgBox.setWindowTitle("提示");
                        msgBox.setText("当前无活动项目，请先在“项目”界面新建或打开一个项目！");
                        msgBox.setIcon(QMessageBox::Warning);
                        msgBox.setStyleSheet(getMessageBoxStyle());
                        msgBox.exec();
                        return;
                    }

                    // 更新按钮选中样式
                    QMap<QString,NavBtn*>::Iterator item = m_NavBtnMap.begin();
                    while (item != m_NavBtnMap.end()) {
                        if(item.key() != name) ((NavBtn*)(item.value()))->setNormalStyle();
                        item++;
                    }

                    // 切换堆叠窗口页面
                    ui->stackedWidget->setCurrentIndex(targetIndex);

                    if (name == tr("图表")) {
                        onTransferDataToPlotting();
                    }
                });
    }

    QSpacerItem* verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    ui->verticalLayoutNav->addSpacerItem(verticalSpacer);

    // 时间显示定时器初始化
    ui->labelTime->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss").replace(" ","\n"));
    connect(&m_timer, &QTimer::timeout, [=] {
        ui->labelTime->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss").replace(" ","\n"));
        ui->labelTime->setStyleSheet("color: black;");
    });
    m_timer.start(1000);

    // --- 初始化各个子页面并挂载到对应容器 ---

    // 1. 项目管理页面
    m_ProjectWidget = new WT_ProjectWidget(ui->pageMonitor);
    ui->verticalLayoutMonitor->addWidget(m_ProjectWidget);
    connect(m_ProjectWidget, &WT_ProjectWidget::projectOpened, this, &MainWindow::onProjectOpened);
    connect(m_ProjectWidget, &WT_ProjectWidget::projectClosed, this, &MainWindow::onProjectClosed);
    connect(m_ProjectWidget, &WT_ProjectWidget::fileLoaded, this, &MainWindow::onFileLoaded);

    // 2. 数据处理页面
    m_DataEditorWidget = new WT_DataWidget(ui->pageHand);
    ui->verticalLayoutHandle->addWidget(m_DataEditorWidget);
    connect(m_DataEditorWidget, &WT_DataWidget::fileChanged, this, &MainWindow::onFileLoaded);
    connect(m_DataEditorWidget, &WT_DataWidget::dataChanged, this, &MainWindow::onDataEditorDataChanged);

    // 3. 图表展示页面
    m_PlottingWidget = new WT_PlottingWidget(ui->pageData);
    ui->verticalLayout_2->addWidget(m_PlottingWidget);
    connect(m_PlottingWidget, &WT_PlottingWidget::viewExportedFile, this, &MainWindow::onViewExportedFile);

    // 4. 试井模型管理初始化
    m_ModelManager = new ModelManager(this);
    m_ModelManager->initializeModels(ui->pageParamter);
    connect(m_ModelManager, &ModelManager::calculationCompleted, this, &MainWindow::onModelCalculationCompleted);

    // 5. 单组数据拟合页面
    if (ui->pageFitting && ui->verticalLayoutFitting) {
        m_FittingPage = new FittingPage(ui->pageFitting);
        ui->verticalLayoutFitting->addWidget(m_FittingPage);
        m_FittingPage->setModelManager(m_ModelManager);
    } else {
        qWarning() << "MainWindow: 拟合界面容器初始化失败";
        m_FittingPage = nullptr;
    }

    // 6. 多数据拟合标签页管理界面（替代原预测页面）
    if (ui->pagePrediction && ui->verticalLayoutPrediction) {
        // 【修改点2】将旧类名 WT_MultidataFittingPage 替换为 multidatafittingpage
        multidatafittingpage* multiPage = new multidatafittingpage(ui->pagePrediction);
        multiPage->setObjectName("MultiDataFittingPage"); // 设定内部名称以便后续查找
        ui->verticalLayoutPrediction->addWidget(multiPage);
        multiPage->setModelManager(m_ModelManager);
    } else {
        qWarning() << "MainWindow: 多数据界面容器初始化失败";
    }

    // 7. 设置页面
    m_SettingsWidget = new SettingsWidget(ui->pageAlarm);
    ui->verticalLayout_3->addWidget(m_SettingsWidget);
    connect(m_SettingsWidget, &SettingsWidget::settingsChanged, this, &MainWindow::onSystemSettingsChanged);

    // 执行各页面的额外初始化
    initProjectForm();
    initDataEditorForm();
    initModelForm();
    initPlottingForm();
    initFittingForm();
    initPredictionForm();
}

// 占位函数：初始化各个界面
void MainWindow::initProjectForm() { qDebug() << "初始化项目界面"; }
void MainWindow::initDataEditorForm() { qDebug() << "初始化数据编辑器界面"; }
void MainWindow::initModelForm() { if (m_ModelManager) qDebug() << "模型界面初始化完成"; }
void MainWindow::initPlottingForm() { qDebug() << "初始化绘图界面"; }
void MainWindow::initFittingForm() { if (m_FittingPage) qDebug() << "拟合界面初始化完成"; }
void MainWindow::initPredictionForm() { qDebug() << "初始化多数据拟合界面完成"; }

// 槽函数：项目打开后触发的数据分发逻辑
void MainWindow::onProjectOpened(bool isNew)
{
    qDebug() << "项目已加载，模式:" << (isNew ? "新建" : "打开");
    m_isProjectLoaded = true;

    if (m_ModelManager) m_ModelManager->updateAllModelsBasicParameters();

    if (m_DataEditorWidget) {
        if (!isNew) m_DataEditorWidget->loadFromProjectData();

        // 传递数据给单组拟合页面
        if (m_FittingPage) m_FittingPage->setProjectDataModels(m_DataEditorWidget->getAllDataModels());

        // 【关键机制】：将项目中的可用数据同步传递给多数据拟合页面标签管理器
        // 【修改点3】同步替换模板参数类名
        multidatafittingpage* multiPage = ui->pagePrediction->findChild<multidatafittingpage*>("MultiDataFittingPage");
        if (multiPage) {
            multiPage->setProjectDataModels(m_DataEditorWidget->getAllDataModels());
        }
    }

    if (m_FittingPage) {
        m_FittingPage->updateBasicParameters();
        m_FittingPage->loadAllFittingStates();
    }

    if (m_PlottingWidget) m_PlottingWidget->loadProjectData();

    updateNavigationState();

    QString title = isNew ? "新建项目成功" : "加载项目成功";
    QString text = isNew ? "新项目已创建。\n基础参数已初始化，您可以开始进行数据录入或模型计算。"
                         : "项目文件加载完成。\n历史参数、数据及图表分析状态已完整恢复。";

    QMessageBox msgBox;
    msgBox.setWindowTitle(title);
    msgBox.setText(text);
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setStyleSheet(getMessageBoxStyle());
    msgBox.exec();
}

// 槽函数：项目关闭后清空并重置所有界面数据
void MainWindow::onProjectClosed()
{
    qDebug() << "项目已关闭，重置界面状态...";
    m_isProjectLoaded = false;
    m_hasValidData = false;

    if (m_DataEditorWidget) m_DataEditorWidget->clearAllData();
    if (m_PlottingWidget) m_PlottingWidget->clearAllPlots();
    if (m_FittingPage) m_FittingPage->resetAnalysis();
    if (m_ModelManager) m_ModelManager->clearCache();
    ModelParameter::instance()->resetAllData();

    // 同时清空多数据标签管理器的缓存数据源
    // 【修改点4】同步替换模板参数类名
    multidatafittingpage* multiPage = ui->pagePrediction->findChild<multidatafittingpage*>("MultiDataFittingPage");
    if (multiPage) {
        multiPage->setProjectDataModels(QMap<QString, QStandardItemModel*>());
    }

    ui->stackedWidget->setCurrentIndex(0);
    updateNavigationState();

    QMessageBox msgBox;
    msgBox.setWindowTitle("提示");
    msgBox.setText("项目已保存并关闭。");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setStyleSheet(getMessageBoxStyle());
    msgBox.exec();
}

// 槽函数：外部文件加载触发界面跳转和数据解析
void MainWindow::onFileLoaded(const QString& filePath, const QString& fileType)
{
    qDebug() << "文件加载：" << filePath;
    if (!m_isProjectLoaded) {
        QMessageBox msgBox;
        msgBox.setWindowTitle("警告");
        msgBox.setText("请先创建或打开项目！");
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setStyleSheet(getMessageBoxStyle());
        msgBox.exec();
        return;
    }

    ui->stackedWidget->setCurrentIndex(1); // 自动跳转到数据页

    QMap<QString,NavBtn*>::Iterator item = m_NavBtnMap.begin();
    while (item != m_NavBtnMap.end()) {
        ((NavBtn*)(item.value()))->setNormalStyle();
        if(item.key() == tr("数据")) ((NavBtn*)(item.value()))->setClickedStyle();
        item++;
    }

    if (m_DataEditorWidget && sender() != m_DataEditorWidget) {
        m_DataEditorWidget->loadData(filePath, fileType);
    }

    // 每次数据源刷新，将最新的数据池也同步派发到对应的拟合页面
    if (m_DataEditorWidget) {
        if (m_FittingPage) {
            m_FittingPage->setProjectDataModels(m_DataEditorWidget->getAllDataModels());
        }

        // 【修改点5】同步替换模板参数类名
        multidatafittingpage* multiPage = ui->pagePrediction->findChild<multidatafittingpage*>("MultiDataFittingPage");
        if (multiPage) {
            multiPage->setProjectDataModels(m_DataEditorWidget->getAllDataModels());
        }
    }

    m_hasValidData = true;
    QTimer::singleShot(1000, this, &MainWindow::onDataReadyForPlotting);
}

// 槽函数：查看导出的文件
void MainWindow::onViewExportedFile(const QString& filePath)
{
    ui->stackedWidget->setCurrentIndex(1);

    QMap<QString,NavBtn*>::Iterator item = m_NavBtnMap.begin();
    while (item != m_NavBtnMap.end()) {
        ((NavBtn*)(item.value()))->setNormalStyle();
        if(item.key() == tr("数据")) ((NavBtn*)(item.value()))->setClickedStyle();
        item++;
    }

    if (m_DataEditorWidget) {
        m_DataEditorWidget->loadData(filePath, "auto");
    }
}

// 槽函数：绘图分析完成事件
void MainWindow::onPlotAnalysisCompleted(const QString &analysisType, const QMap<QString, double> &results) {
    qDebug() << "绘图分析完成：" << analysisType;
}

// 槽函数：数据准备完成事件，转交数据给绘图模块
void MainWindow::onDataReadyForPlotting() {
    transferDataFromEditorToPlotting();
}

// 槽函数：处理传输数据到图表界面的请求
void MainWindow::onTransferDataToPlotting()
{
    if (!hasDataLoaded()) return;
    transferDataFromEditorToPlotting();
}

// 槽函数：数据编辑器中的数据发生变动时同步绘图界面
void MainWindow::onDataEditorDataChanged()
{
    if (ui->stackedWidget->currentIndex() == 2) {
        transferDataFromEditorToPlotting();
    }
    m_hasValidData = hasDataLoaded();
}

// 槽函数：模型计算完成
void MainWindow::onModelCalculationCompleted(const QString &analysisType, const QMap<QString, double> &results)
{
    qDebug() << "模型计算完成：" << analysisType;
}

// 槽函数：将当前活动的观测数据计算导数后传输给拟合模块
void MainWindow::transferDataToFitting()
{
    if (!m_FittingPage || !m_DataEditorWidget) return;

    QStandardItemModel* model = m_DataEditorWidget->getDataModel();
    if (!model || model->rowCount() == 0) return;

    QVector<double> tVec, pVec, dVec;
    double p_initial = 0.0;

    for(int r = 0; r < model->rowCount(); ++r) {
        double p = model->index(r, 1).data().toDouble();
        if (std::abs(p) > 1e-6) { p_initial = p; break; }
    }

    for(int r = 0; r < model->rowCount(); ++r) {
        double t = model->index(r, 0).data().toDouble();
        double p_raw = model->index(r, 1).data().toDouble();
        if (t > 0) {
            tVec.append(t);
            pVec.append(std::abs(p_raw - p_initial));
        }
    }

    if (tVec.size() > 2) {
        dVec = PressureDerivativeCalculator::calculateBourdetDerivative(tVec, pVec, 0.1);
    } else {
        dVec.resize(tVec.size());
        dVec.fill(0.0);
    }

    m_FittingPage->setObservedDataToCurrent(tVec, pVec, dVec);
}

// 槽函数：响应拟合进度条更新并在状态栏显示
void MainWindow::onFittingProgressChanged(int progress)
{
    if (this->statusBar()) {
        this->statusBar()->showMessage(QString("正在拟合... %1%").arg(progress));
        if(progress >= 100) this->statusBar()->showMessage("拟合完成", 5000);
    }
}

// 槽函数：系统设置改变
void MainWindow::onSystemSettingsChanged() { qDebug() << "系统设置已变更"; }

// 槽函数：性能设置改变
void MainWindow::onPerformanceSettingsChanged() {}

// 辅助函数：获取数据编辑器模型
QStandardItemModel* MainWindow::getDataEditorModel() const
{
    if (!m_DataEditorWidget) return nullptr;
    return m_DataEditorWidget->getDataModel();
}

// 辅助函数：获取当前文件名
QString MainWindow::getCurrentFileName() const
{
    if (!m_DataEditorWidget) return QString();
    return m_DataEditorWidget->getCurrentFileName();
}

// 辅助函数：检查是否加载了数据
bool MainWindow::hasDataLoaded()
{
    if (!m_DataEditorWidget) return false;
    return m_DataEditorWidget->hasData();
}

// 辅助函数：提取所有数据编辑器中的数据到绘图页面
void MainWindow::transferDataFromEditorToPlotting()
{
    if (!m_DataEditorWidget || !m_PlottingWidget) return;
    QMap<QString, QStandardItemModel*> models = m_DataEditorWidget->getAllDataModels();
    m_PlottingWidget->setDataModels(models);
    if (!models.isEmpty()) m_hasValidData = true;
}

// 辅助函数：更新导航按钮高亮状态为项目页
void MainWindow::updateNavigationState()
{
    QMap<QString,NavBtn*>::Iterator item = m_NavBtnMap.begin();
    while (item != m_NavBtnMap.end()) {
        ((NavBtn*)(item.value()))->setNormalStyle();
        if(item.key() == tr("项目")) ((NavBtn*)(item.value()))->setClickedStyle();
        item++;
    }
}

// 辅助函数：获取统一MessageBox样式
QString MainWindow::getMessageBoxStyle() const
{
    return getGlobalMessageBoxStyle();
}
