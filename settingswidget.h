/*
 * settingswidget.h
 * 文件作用：系统设置窗口的头文件
 * 功能描述：
 * 1. 定义 SettingsWidget 类，继承自 QWidget
 * 2. 声明各个设置模块（通用、单位、绘图、路径、系统）的 UI 组件交互逻辑
 * 3. 声明配置数据的加载 (load)、保存 (apply) 和恢复默认 (restoreDefaults) 方法
 * 4. 定义配置变更的信号，供主程序响应（如切换单位、修改绘图风格）
 */

#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QWidget>
#include <QSettings>
#include <QFileDialog>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>
#include <QTimer>

namespace Ui {
class SettingsWidget;
}

class SettingsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SettingsWidget(QWidget *parent = nullptr);
    ~SettingsWidget();

    // --- 公共接口：获取当前配置值 ---

    // 路径配置
    QString getDataPath() const;
    QString getReportPath() const;
    QString getBackupPath() const;

    // 系统配置
    int getAutoSaveInterval() const;
    bool isBackupEnabled() const;

    // 单位配置 [新增]
    int getPressureUnitIndex() const; // 0: MPa, 1: psi
    int getRateUnitIndex() const;     // 0: m3/d, 1: bbl/d
    int getPrecision() const;         // 小数位数

    // 绘图配置 [新增]
    int getPlotBackgroundStyle() const; // 0: 白色, 1: 深色
    bool isGridVisibleDefault() const;

signals:
    // 配置变更信号
    void settingsChanged();           // 通用变更信号
    void themeChanged(int themeIdx);  // 主题变更
    void unitSystemChanged();         // 单位制变更
    void plotStyleChanged();          // 绘图风格变更

private slots:
    // 侧边导航栏切换
    void on_navTupleList_currentRowChanged(int currentRow);

    // 路径浏览按钮
    void on_btnBrowseData_clicked();
    void on_btnBrowseReport_clicked();
    void on_btnBrowseBackup_clicked();

    // 底部操作按钮
    void on_btnRestoreDefaults_clicked(); // 恢复默认
    void on_btnApply_clicked();           // 应用保存
    void on_btnCancel_clicked();          // 取消/关闭

    // 监听部分控件变化（用于激活"应用"按钮状态等）
    void onSettingModified();

private:
    Ui::SettingsWidget *ui;
    QSettings *m_settings;
    bool m_isModified; // 记录是否有未保存的修改

    // --- 核心逻辑方法 ---

    // 初始化界面控件（设置下拉框选项、默认值等）
    void initInterface();

    // 加载所有设置到 UI
    void loadSettings();

    // 保存 UI 设置到配置文件
    void applySettings();

    // 恢复所有设置到出厂默认值
    void restoreDefaults();

    // 路径有效性验证
    bool validatePaths();

    // 辅助：获取默认路径
    QString getDefaultPath(const QString &subDir);

    // 辅助：创建目录
    void ensureDirExists(const QString &path);

    // 常量定义
    static const int DEFAULT_AUTO_SAVE;
    static const int DEFAULT_MAX_BACKUPS;
};

#endif // SETTINGSWIDGET_H
