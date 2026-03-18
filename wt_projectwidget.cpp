/*
 * wt_projectwidget.cpp
 * 文件作用：项目管理界面实现文件
 * 功能描述：
 * 1. 初始化界面样式和按钮事件连接。
 * 2. 实现"新建"、"打开"、"关闭"、"退出"的详细交互逻辑。
 * 3. [修复] 正确传递新增的水平井长度和裂缝条数参数，解决了之前的编译错误。
 */

#include "wt_projectwidget.h"
#include "ui_wt_projectwidget.h"
#include "newprojectdialog.h"
#include "modelparameter.h" // 全局参数管理类

#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QPalette>
#include <QFileInfo>
#include <QPushButton> // 必须包含，用于比较 clickedButton

WT_ProjectWidget::WT_ProjectWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WT_ProjectWidget),
    m_isProjectOpen(false),         // 初始状态：无项目打开
    m_currentProjectFilePath("")    // 初始路径为空
{
    ui->setupUi(this);
    init();
}

WT_ProjectWidget::~WT_ProjectWidget()
{
    delete ui;
}

void WT_ProjectWidget::init()
{
    qDebug() << "初始化项目管理界面...";

    // --- 界面样式初始化开始 ---
    // 设置透明背景，适应整体UI风格
    this->setStyleSheet("background-color: transparent;");
    ui->widget_5->setStyleSheet("background-color: transparent;");

    // 调整网格布局间距
    ui->gridLayout_3->setHorizontalSpacing(30);
    ui->gridLayout_3->setVerticalSpacing(10);

    // 定义通用字体
    QFont bigFont;
    bigFont.setPointSize(16);
    bigFont.setBold(true);

    // 定义按钮背景色
    QColor backgroundColor(148, 226, 255);

    // 定义通用样式表
    QString forceStyle = QString(
        "MonitoStateW { "
        "background-color: rgb(148, 226, 255); "
        "border-radius: 10px; "
        "padding: 10px; "
        "} "
        "MonitoStateW * { "
        "background-color: transparent; "
        "} "
        "MonitoStateW:hover { "
        "background-color: rgb(120, 200, 240); "
        "} "
        "QLabel { "
        "color: #333333; "
        "font-weight: bold; "
        "margin-top: 5px; "
        "background-color: transparent; "
        "}"
        );

    // 通用顶部样式
    QString topPicStyle = "";
    QString topName = "  ";

    // 1. 配置 "新建" 按钮
    QString centerPicStyle1 = "border-image: url(:/new/prefix1/Resource/PRO1.png);";
    ui->MonitState1->setTextInfo(centerPicStyle1, topPicStyle, topName, "新建");
    ui->MonitState1->setFixedSize(128, 160);
    ui->MonitState1->setStyleSheet(forceStyle);
    ui->MonitState1->setAutoFillBackground(true);
    QPalette pal1 = ui->MonitState1->palette(); pal1.setColor(QPalette::Window, backgroundColor); ui->MonitState1->setPalette(pal1);
    ui->MonitState1->setFont(bigFont);
    connect(ui->MonitState1, SIGNAL(sigClicked()), this, SLOT(onNewProjectClicked()));

    // 2. 配置 "打开" 按钮
    QString centerPicStyle2 = "border-image: url(:/new/prefix1/Resource/PRO2.png);";
    ui->MonitState2->setTextInfo(centerPicStyle2, topPicStyle, topName, "打开");
    ui->MonitState2->setFixedSize(128, 160);
    ui->MonitState2->setStyleSheet(forceStyle);
    ui->MonitState2->setAutoFillBackground(true);
    QPalette pal2 = ui->MonitState2->palette(); pal2.setColor(QPalette::Window, backgroundColor); ui->MonitState2->setPalette(pal2);
    ui->MonitState2->setFont(bigFont);
    connect(ui->MonitState2, SIGNAL(sigClicked()), this, SLOT(onOpenProjectClicked()));

    // 3. 配置 "关闭" 按钮
    QString centerPicStyle3 = "border-image: url(:/new/prefix1/Resource/PRO3.png);";
    ui->MonitState3->setTextInfo(centerPicStyle3, topPicStyle, topName, "关闭");
    ui->MonitState3->setFixedSize(128, 160);
    ui->MonitState3->setStyleSheet(forceStyle);
    ui->MonitState3->setAutoFillBackground(true);
    QPalette pal3 = ui->MonitState3->palette(); pal3.setColor(QPalette::Window, backgroundColor); ui->MonitState3->setPalette(pal3);
    ui->MonitState3->setFont(bigFont);
    connect(ui->MonitState3, SIGNAL(sigClicked()), this, SLOT(onCloseProjectClicked()));

    // 4. 配置 "退出" 按钮
    QString centerPicStyle4 = "border-image: url(:/new/prefix1/Resource/PRO4.png);";
    ui->MonitState4->setTextInfo(centerPicStyle4, topPicStyle, topName, "退出");
    ui->MonitState4->setFixedSize(128, 160);
    ui->MonitState4->setStyleSheet(forceStyle);
    ui->MonitState4->setAutoFillBackground(true);
    QPalette pal4 = ui->MonitState4->palette(); pal4.setColor(QPalette::Window, backgroundColor); ui->MonitState4->setPalette(pal4);
    ui->MonitState4->setFont(bigFont);
    connect(ui->MonitState4, SIGNAL(sigClicked()), this, SLOT(onExitClicked()));
}

void WT_ProjectWidget::setProjectState(bool isOpen, const QString& filePath)
{
    m_isProjectOpen = isOpen;
    m_currentProjectFilePath = filePath;
    qDebug() << "项目状态更新: 打开=" << isOpen << " 路径=" << filePath;
}

// 获取统一的白底黑字弹窗样式
QString WT_ProjectWidget::getMessageBoxStyle() const
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

// ============================================================================
// 按钮槽函数实现
// ============================================================================

// 1. 点击“新建”按钮
void WT_ProjectWidget::onNewProjectClicked()
{
    qDebug() << "点击了[新建]按钮";

    // 逻辑：如果已有项目打开，阻止新建，提示先关闭
    if (m_isProjectOpen) {
        QString projName = QFileInfo(m_currentProjectFilePath).fileName();
        if(projName.isEmpty()) projName = "当前项目";

        QMessageBox msgBox;
        msgBox.setWindowTitle("操作受限");
        msgBox.setText(QString("项目 [%1] 正在运行中。\n为了数据安全，不能直接创建新项目。\n请先点击“关闭”按钮关闭当前项目。").arg(projName));
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setStyleSheet(getMessageBoxStyle()); // 应用样式
        msgBox.exec();
        return;
    }

    // 正常流程：弹出新建对话框
    NewProjectDialog* dialog = new NewProjectDialog(this);
    if (dialog->exec() == QDialog::Accepted) {
        ProjectData data = dialog->getProjectData();

        // [关键修复] 参数必须完整，顺序必须与 ModelParameter::setParameters 一致
        ModelParameter::instance()->setParameters(
            data.porosity,          // phi
            data.thickness,         // h
            data.viscosity,         // mu
            data.volumeFactor,      // B
            data.compressibility,   // Ct
            data.productionRate,    // q
            data.wellRadius,        // rw
            data.horizLength,       // L (新增)
            data.fracCount,         // nf (新增)
            data.fullFilePath       // path
            );

        // 设置状态为已打开
        setProjectState(true, data.fullFilePath);

        // 发送信号通知主界面 (主界面会弹出唯一的成功提示)
        emit projectOpened(true);
    }
    delete dialog;
}

// 2. 点击“打开”按钮
void WT_ProjectWidget::onOpenProjectClicked()
{
    qDebug() << "点击了[打开]按钮";

    // 逻辑：如果已有项目打开，阻止打开新文件
    if (m_isProjectOpen) {
        QString projName = QFileInfo(m_currentProjectFilePath).fileName();
        if(projName.isEmpty()) projName = "当前项目";

        QMessageBox msgBox;
        msgBox.setWindowTitle("操作受限");
        msgBox.setText(QString("项目 [%1] 已经打开。\n不能同时打开多个项目。\n请先点击“关闭”按钮关闭当前项目。").arg(projName));
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setStyleSheet(getMessageBoxStyle()); // 应用样式
        msgBox.exec();
        return;
    }

    // 正常流程：选择文件并打开
    QString filePath = QFileDialog::getOpenFileName(
        this,
        tr("打开项目"),
        "",
        tr("WellTest Project (*.pwt)")
        );

    if (filePath.isEmpty()) return;

    // 加载项目数据
    if (ModelParameter::instance()->loadProject(filePath)) {
        // 设置状态为已打开
        setProjectState(true, filePath);

        // 发送信号通知主界面 (主界面会弹出唯一的成功提示)
        emit projectOpened(false);
    } else {
        // 失败情况保留弹窗
        QMessageBox msgBox;
        msgBox.setWindowTitle("错误");
        msgBox.setText("项目文件损坏或格式不正确，无法打开。");
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setStyleSheet(getMessageBoxStyle());
        msgBox.exec();
    }
}

// 3. 点击“关闭”按钮
void WT_ProjectWidget::onCloseProjectClicked()
{
    qDebug() << "点击了[关闭]按钮";

    // 逻辑：如果没有项目打开，提示错误
    if (!m_isProjectOpen) {
        QMessageBox msgBox;
        msgBox.setWindowTitle("提示");
        msgBox.setText("当前没有正在运行的项目，无法执行关闭操作。");
        msgBox.setIcon(QMessageBox::Information);
        msgBox.setStyleSheet(getMessageBoxStyle()); // 应用样式
        msgBox.exec();
        return;
    }

    // 逻辑：有项目打开，弹出三选一对话框
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("关闭项目");
    msgBox.setText(QString("是否关闭当前项目 [%1]？").arg(QFileInfo(m_currentProjectFilePath).fileName()));
    msgBox.setInformativeText("关闭前建议保存数据。");
    msgBox.setIcon(QMessageBox::Question);
    msgBox.setStyleSheet(getMessageBoxStyle()); // 应用样式

    // 自定义按钮
    QPushButton *saveCloseBtn = msgBox.addButton("保存并关闭", QMessageBox::AcceptRole);
    QPushButton *directCloseBtn = msgBox.addButton("直接关闭", QMessageBox::DestructiveRole);
    msgBox.addButton("取消", QMessageBox::RejectRole); // 仅添加，不需保存变量

    msgBox.setDefaultButton(saveCloseBtn);
    msgBox.exec();

    if (msgBox.clickedButton() == saveCloseBtn) {
        // 选项1：保存并关闭
        if (saveCurrentProject()) {
            closeProjectInternal();
        }
    } else if (msgBox.clickedButton() == directCloseBtn) {
        // 选项2：直接关闭
        closeProjectInternal();
    }
}

// 4. 点击“退出”按钮
void WT_ProjectWidget::onExitClicked()
{
    qDebug() << "点击了[退出]按钮";

    // 逻辑：如果没有项目打开，直接退出
    if (!m_isProjectOpen) {
        QApplication::quit();
        return;
    }

    // 逻辑：有项目打开，确认是否保存
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("退出系统");
    msgBox.setText("当前有项目正在运行，确定要退出吗？");
    msgBox.setInformativeText("建议在退出前保存当前项目。");
    msgBox.setIcon(QMessageBox::Question);
    msgBox.setStyleSheet(getMessageBoxStyle()); // 应用样式

    QPushButton *saveExitBtn = msgBox.addButton("保存并退出", QMessageBox::YesRole);
    QPushButton *directExitBtn = msgBox.addButton("直接退出", QMessageBox::NoRole);
    msgBox.addButton("取消", QMessageBox::RejectRole);

    msgBox.exec();

    if (msgBox.clickedButton() == saveExitBtn) {
        // 保存后退出
        saveCurrentProject();
        QApplication::quit();
    } else if (msgBox.clickedButton() == directExitBtn) {
        // 直接退出
        QApplication::quit();
    }
}

// 备用：文件读取逻辑
void WT_ProjectWidget::onLoadFileClicked()
{
    QString filter = tr("Excel Files (*.xlsx *.xls);;Text Files (*.txt);;All Files (*.*)");
    QString filePath = QFileDialog::getOpenFileName(this, tr("选择要读取的数据文件"), QString(), filter);

    if (filePath.isEmpty()) return;

    QString fileType = "unknown";
    if (filePath.endsWith(".xlsx", Qt::CaseInsensitive) || filePath.endsWith(".xls", Qt::CaseInsensitive)) {
        fileType = "excel";
    } else if (filePath.endsWith(".txt", Qt::CaseInsensitive)) {
        fileType = "txt";
    }

    emit fileLoaded(filePath, fileType);

    QMessageBox msgBox;
    msgBox.setWindowTitle(tr("文件读取"));
    msgBox.setText(tr("文件已成功读取，正在准备显示数据..."));
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setStyleSheet(getMessageBoxStyle());
    msgBox.exec();
}

// ============================================================================
// 私有辅助函数
// ============================================================================

bool WT_ProjectWidget::saveCurrentProject()
{
    qDebug() << "正在保存项目:" << m_currentProjectFilePath;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    // ModelParameter::instance()->saveProject(m_currentProjectFilePath);
    QApplication::restoreOverrideCursor();
    return true;
}

void WT_ProjectWidget::closeProjectInternal()
{
    // 重置状态
    setProjectState(false, "");

    // 通知主界面
    emit projectClosed();
}
