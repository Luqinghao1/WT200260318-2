/*
 * 文件名: mainwindow.h
 * 文件作用: 主窗口类头文件
 * 功能描述:
 * 1. 声明主窗口框架及各个子功能模块指针。
 * 2. 引入 ModelManager 头文件以访问模型系统。
 * 3. 定义主窗口与各个子模块（项目、数据、绘图、拟合）之间的交互接口。
 * 4. [新增] 增加了 onViewExportedFile 槽函数，处理从图表导出的文件跳转。
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMap>
#include <QTimer>
#include <QStandardItemModel>
#include "modelmanager.h"

// 前置声明各个功能页面的类
class NavBtn;
class WT_ProjectWidget;
class WT_DataWidget;
class WT_PlottingWidget;
class FittingPage;
class SettingsWidget;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // 初始化主程序逻辑
    void init();

    // 各功能模块界面初始化函数
    void initProjectForm();     // 初始化项目管理界面
    void initDataEditorForm();  // 初始化数据编辑界面
    void initModelForm();       // 初始化模型参数界面
    void initPlottingForm();    // 初始化图表分析界面
    void initFittingForm();     // 初始化拟合分析界面
    void initPredictionForm();  // 初始化产能预测界面 (预留)

private slots:
    // --- 项目管理相关槽函数 ---
    void onProjectOpened(bool isNew);  // 项目打开或新建成功后触发
    void onProjectClosed();            // 项目关闭后触发
    void onFileLoaded(const QString& filePath, const QString& fileType); // 外部文件加载后触发

    // --- 数据交互与分析相关槽函数 ---
    void onPlotAnalysisCompleted(const QString &analysisType, const QMap<QString, double> &results); // 绘图分析完成
    void onDataReadyForPlotting();         // 数据准备好进行绘图
    void onTransferDataToPlotting();       // 请求将数据传输给绘图模块
    void onDataEditorDataChanged();        // 数据编辑器中数据发生变化

    // [新增] 处理从图表界面导出的文件查看请求
    void onViewExportedFile(const QString& filePath);

    // --- 设置与计算相关槽函数 ---
    void onSystemSettingsChanged();        // 系统设置变更
    void onPerformanceSettingsChanged();   // 性能设置变更
    void onModelCalculationCompleted(const QString &analysisType, const QMap<QString, double> &results); // 模型计算完成
    void onFittingProgressChanged(int progress); // 拟合进度更新

private:
    Ui::MainWindow *ui;

    // 各个功能页面的指针成员变量
    WT_ProjectWidget* m_ProjectWidget;      // 项目管理页
    WT_DataWidget* m_DataEditorWidget;      // 数据编辑页 (支持多标签)
    ModelManager* m_ModelManager;           // 模型参数页
    WT_PlottingWidget* m_PlottingWidget;    // 图表分析页
    FittingPage* m_FittingPage;             // 拟合分析页
    SettingsWidget* m_SettingsWidget;       // 系统设置页

    QMap<QString, NavBtn*> m_NavBtnMap;     // 左侧导航按钮映射表
    QTimer m_timer;                         // 系统时间显示定时器
    bool m_hasValidData = false;            // 标记当前是否有有效数据
    bool m_isProjectLoaded = false;         // 标记项目是否已加载

    // --- 内部辅助函数 ---

    // 将数据编辑器中的所有数据传输给绘图模块 (Plotting)
    void transferDataFromEditorToPlotting();

    // 更新左侧导航栏的选中状态
    void updateNavigationState();

    // [保留接口] 将当前活动的观测数据传输给拟合模块
    void transferDataToFitting();

    // 获取当前活动的数据模型 (单个)
    QStandardItemModel* getDataEditorModel() const;

    // 获取当前活动文件的名称
    QString getCurrentFileName() const;

    // 检查是否有数据加载
    bool hasDataLoaded();

    // 获取全局统一的消息框样式表
    QString getMessageBoxStyle() const;
};

#endif // MAINWINDOW_H
