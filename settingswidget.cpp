/*
 * settingswidget.cpp
 * 文件作用：系统设置窗口的具体实现
 * 功能描述：
 * 1. 构造函数初始化 UI、连接信号槽、加载 QSettings 配置
 * 2. 实现五个功能模块（通用、单位、绘图、路径、系统）的具体的加载与保存逻辑
 * 3. 实现路径选择对话框的弹出与回填
 * 4. 实现“恢复默认值”逻辑，重置所有控件状态
 */

#include "settingswidget.h"
#include "ui_settingswidget.h"
#include <QDebug>
#include <QDate>

// 默认常量定义
const int SettingsWidget::DEFAULT_AUTO_SAVE = 10;
const int SettingsWidget::DEFAULT_MAX_BACKUPS = 10;

SettingsWidget::SettingsWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SettingsWidget),
    m_settings(nullptr),
    m_isModified(false)
{
    ui->setupUi(this);

    // 初始化 QSettings (组织名, 应用名)
    m_settings = new QSettings("WellTestPro", "WellTestAnalysis", this);

    // 初始化界面控件内容
    initInterface();

    // 加载已保存的设置
    loadSettings();

    // 连接通用修改信号（用于监控数据变更）
    // 这里简单处理，任何输入框或复选框变化都触发 onSettingModified
    QList<QWidget*> widgets = this->findChildren<QWidget*>();
    for(auto w : widgets) {
        if(qobject_cast<QLineEdit*>(w))
            connect(qobject_cast<QLineEdit*>(w), &QLineEdit::textChanged, this, &SettingsWidget::onSettingModified);
        else if(qobject_cast<QSpinBox*>(w))
            connect(qobject_cast<QSpinBox*>(w), QOverload<int>::of(&QSpinBox::valueChanged), this, &SettingsWidget::onSettingModified);
        else if(qobject_cast<QComboBox*>(w))
            connect(qobject_cast<QComboBox*>(w), QOverload<int>::of(&QComboBox::currentIndexChanged), this, &SettingsWidget::onSettingModified);
        else if(qobject_cast<QCheckBox*>(w))
            connect(qobject_cast<QCheckBox*>(w), &QCheckBox::toggled, this, &SettingsWidget::onSettingModified);
    }
}

SettingsWidget::~SettingsWidget()
{
    delete ui;
}

void SettingsWidget::initInterface()
{
    // 1. 设置导航栏图标（假设资源文件中有 Nav1-Nav5，若无则只显示文字）
    // 为了代码健壮性，这里暂时只设置默认行，图标在UI设计器中已指定或通过样式表控制
    ui->navTupleList->setCurrentRow(0); // 默认选中第一页

    // 2. 初始化单位下拉框
    ui->cmbPressureUnit->clear();
    ui->cmbPressureUnit->addItems({"MPa (兆帕)", "psi (磅/平方英寸)", "bar (巴)"});

    ui->cmbRateUnit->clear();
    ui->cmbRateUnit->addItems({"m³/d (立方米/天)", "bbl/d (桶/天)", "t/d (吨/天)"});

    // 3. 初始化绘图背景
    ui->cmbPlotBackground->clear();
    ui->cmbPlotBackground->addItems({"白色主题 (默认)", "深色主题 (护眼)", "灰色网格"});

    // 4. 初始化日志级别
    ui->cmbLogLevel->clear();
    ui->cmbLogLevel->addItems({"仅错误 (Error)", "警告与错误 (Warning)", "一般信息 (Info)", "详细调试 (Debug)"});
}

void SettingsWidget::loadSettings()
{
    // --- 1. 通用设置 ---
    ui->cmbTheme->setCurrentIndex(m_settings->value("general/theme", 0).toInt());
    ui->chkStartFullScreen->setChecked(m_settings->value("general/fullScreen", false).toBool());

    // --- 2. 单位与精度 ---
    ui->cmbPressureUnit->setCurrentIndex(m_settings->value("units/pressure", 0).toInt()); // 默认 MPa
    ui->cmbRateUnit->setCurrentIndex(m_settings->value("units/rate", 0).toInt());         // 默认 m3/d
    ui->spinPrecision->setValue(m_settings->value("units/precision", 4).toInt());         // 默认 4位小数

    // --- 3. 绘图设置 ---
    ui->cmbPlotBackground->setCurrentIndex(m_settings->value("plot/background", 0).toInt());
    ui->chkShowGrid->setChecked(m_settings->value("plot/showGrid", true).toBool());
    ui->spinLineWidth->setValue(m_settings->value("plot/lineWidth", 2).toInt());

    // --- 4. 路径设置 ---
    QString docPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    ui->lineDataPath->setText(m_settings->value("paths/data", docPath + "/WellTestPro/Data").toString());
    ui->lineReportPath->setText(m_settings->value("paths/report", docPath + "/WellTestPro/Reports").toString());
    ui->lineBackupPath->setText(m_settings->value("paths/backup", docPath + "/WellTestPro/Backups").toString());

    // --- 5. 系统设置 ---
    ui->spinAutoSave->setValue(m_settings->value("system/autoSaveInterval", DEFAULT_AUTO_SAVE).toInt());
    ui->chkEnableBackup->setChecked(m_settings->value("system/backupEnabled", true).toBool());
    ui->spinMaxBackups->setValue(m_settings->value("system/maxBackups", DEFAULT_MAX_BACKUPS).toInt());
    ui->chkCleanupLogs->setChecked(m_settings->value("system/cleanupLogs", true).toBool());
    ui->spinLogDays->setValue(m_settings->value("system/logRetention", 30).toInt());
    ui->cmbLogLevel->setCurrentIndex(m_settings->value("system/logLevel", 2).toInt());

    m_isModified = false;
}

void SettingsWidget::applySettings()
{
    if (!validatePaths()) {
        QMessageBox::warning(this, "路径错误", "配置的路径不能为空且必须具有读写权限！");
        return;
    }

    // 保存所有配置
    m_settings->setValue("general/theme", ui->cmbTheme->currentIndex());
    m_settings->setValue("general/fullScreen", ui->chkStartFullScreen->isChecked());

    m_settings->setValue("units/pressure", ui->cmbPressureUnit->currentIndex());
    m_settings->setValue("units/rate", ui->cmbRateUnit->currentIndex());
    m_settings->setValue("units/precision", ui->spinPrecision->value());

    m_settings->setValue("plot/background", ui->cmbPlotBackground->currentIndex());
    m_settings->setValue("plot/showGrid", ui->chkShowGrid->isChecked());
    m_settings->setValue("plot/lineWidth", ui->spinLineWidth->value());

    m_settings->setValue("paths/data", ui->lineDataPath->text());
    m_settings->setValue("paths/report", ui->lineReportPath->text());
    m_settings->setValue("paths/backup", ui->lineBackupPath->text());

    m_settings->setValue("system/autoSaveInterval", ui->spinAutoSave->value());
    m_settings->setValue("system/backupEnabled", ui->chkEnableBackup->isChecked());
    m_settings->setValue("system/maxBackups", ui->spinMaxBackups->value());
    m_settings->setValue("system/cleanupLogs", ui->chkCleanupLogs->isChecked());
    m_settings->setValue("system/logRetention", ui->spinLogDays->value());
    m_settings->setValue("system/logLevel", ui->cmbLogLevel->currentIndex());

    m_settings->sync(); // 强制写入磁盘

    // 发射信号通知系统其他部分
    emit settingsChanged();
    emit unitSystemChanged();
    emit plotStyleChanged();

    QMessageBox::information(this, "系统设置", "设置已保存并生效！");
    m_isModified = false;

    // 确保目录存在
    ensureDirExists(ui->lineDataPath->text());
    ensureDirExists(ui->lineReportPath->text());
    ensureDirExists(ui->lineBackupPath->text());
}

void SettingsWidget::restoreDefaults()
{
    if(QMessageBox::question(this, "确认重置", "确定要将所有设置恢复为出厂默认值吗？\n此操作不可撤销。") != QMessageBox::Yes) {
        return;
    }

    // 清除所有保存的设置
    m_settings->clear();
    m_settings->sync();

    // 重新加载（因为 clear 了，loadSettings 会读取代码中写的默认值）
    loadSettings();

    QMessageBox::information(this, "系统设置", "已恢复默认设置。");
}

bool SettingsWidget::validatePaths()
{
    if(ui->lineDataPath->text().isEmpty()) return false;
    if(ui->lineReportPath->text().isEmpty()) return false;
    if(ui->lineBackupPath->text().isEmpty()) return false;
    return true;
}

void SettingsWidget::ensureDirExists(const QString &path)
{
    QDir dir(path);
    if (!dir.exists()) {
        dir.mkpath(path);
    }
}

// 槽函数：导航切换
void SettingsWidget::on_navTupleList_currentRowChanged(int currentRow)
{
    // QStackedWidget 的索引与 ListWidget 行号对应
    ui->stackedContent->setCurrentIndex(currentRow);

    // 更新标题栏文字
    QStringList titles = {
        "通用设置 - 界面与启动选项",
        "单位与精度 - 物理量单位配置",
        "绘图设置 - 图表默认风格",
        "路径配置 - 文件存储位置",
        "系统与日志 - 运行维护设置"
    };
    if(currentRow >= 0 && currentRow < titles.size())
        ui->lblPageTitle->setText(titles[currentRow]);
}

// 槽函数：浏览按钮
void SettingsWidget::on_btnBrowseData_clicked() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择数据存储路径", ui->lineDataPath->text());
    if(!dir.isEmpty()) ui->lineDataPath->setText(dir);
}
void SettingsWidget::on_btnBrowseReport_clicked() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择报告输出路径", ui->lineReportPath->text());
    if(!dir.isEmpty()) ui->lineReportPath->setText(dir);
}
void SettingsWidget::on_btnBrowseBackup_clicked() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择备份路径", ui->lineBackupPath->text());
    if(!dir.isEmpty()) ui->lineBackupPath->setText(dir);
}

// 槽函数：底部按钮
void SettingsWidget::on_btnRestoreDefaults_clicked() {
    restoreDefaults();
}
void SettingsWidget::on_btnApply_clicked() {
    applySettings();
}
void SettingsWidget::on_btnCancel_clicked() {
    // 如果有修改未保存，可以提示（此处简单处理为直接关闭）
    this->close();
}
void SettingsWidget::onSettingModified() {
    m_isModified = true;
}

// Getters implementation
QString SettingsWidget::getDataPath() const { return ui->lineDataPath->text(); }
QString SettingsWidget::getReportPath() const { return ui->lineReportPath->text(); }
QString SettingsWidget::getBackupPath() const { return ui->lineBackupPath->text(); }
int SettingsWidget::getAutoSaveInterval() const { return ui->spinAutoSave->value(); }
bool SettingsWidget::isBackupEnabled() const { return ui->chkEnableBackup->isChecked(); }
int SettingsWidget::getPressureUnitIndex() const { return ui->cmbPressureUnit->currentIndex(); }
int SettingsWidget::getRateUnitIndex() const { return ui->cmbRateUnit->currentIndex(); }
int SettingsWidget::getPrecision() const { return ui->spinPrecision->value(); }
int SettingsWidget::getPlotBackgroundStyle() const { return ui->cmbPlotBackground->currentIndex(); }
bool SettingsWidget::isGridVisibleDefault() const { return ui->chkShowGrid->isChecked(); }
