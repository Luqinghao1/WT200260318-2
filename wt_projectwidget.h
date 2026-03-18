/*
 * wt_projectwidget.h
 * 文件作用：项目管理界面头文件
 * 功能描述：
 * 1. 定义项目管理界面的类结构，包含新建、打开、关闭、退出四个主要功能入口。
 * 2. 维护当前项目的打开状态 (m_isProjectOpen) 和项目路径信息。
 * 3. 声明各个按钮点击后的槽函数，实现基于状态的交互逻辑判断。
 * 4. 提供统一的弹窗样式获取函数。
 */

#ifndef WT_PROJECTWIDGET_H
#define WT_PROJECTWIDGET_H

#include <QWidget>
#include <QString>

// 前向声明 UI 类
namespace Ui {
class WT_ProjectWidget;
}

class WT_ProjectWidget : public QWidget
{
    Q_OBJECT

public:
    // 构造函数
    explicit WT_ProjectWidget(QWidget *parent = nullptr);
    // 析构函数
    ~WT_ProjectWidget();

    // 初始化界面UI和信号连接
    void init();

    // 设置当前项目状态（供外部或内部加载成功后调用）
    // isOpen: 是否有项目打开
    // filePath: 项目文件路径（用于显示名称和保存）
    void setProjectState(bool isOpen, const QString& filePath = "");

signals:
    // 信号：新项目创建或打开成功 (通知主界面解锁功能)
    void projectOpened(bool isNew);

    // 信号：项目已关闭 (通知主界面重置状态)
    void projectClosed();

    // 信号：请求加载文件 (用于导入数据文件，保留原有功能)
    void fileLoaded(const QString& filePath, const QString& fileType);

private slots:
    // 槽函数：点击"新建"按钮
    void onNewProjectClicked();

    // 槽函数：点击"打开"按钮
    void onOpenProjectClicked();

    // 槽函数：点击"关闭"按钮
    void onCloseProjectClicked();

    // 槽函数：点击"退出"按钮
    void onExitClicked();

    // 槽函数：点击"读取"按钮 (备用功能)
    void onLoadFileClicked();

private:
    Ui::WT_ProjectWidget *ui;

    // 核心状态变量：当前是否有项目打开
    bool m_isProjectOpen;

    // 变量：当前打开的项目文件完整路径
    QString m_currentProjectFilePath;

    // 辅助函数：执行保存项目的逻辑
    // 返回值 true 表示保存成功，false 表示失败
    bool saveCurrentProject();

    // 辅助函数：执行关闭项目的清理逻辑（不弹窗，只发信号）
    void closeProjectInternal();

    // 辅助函数：获取统一的弹窗样式表（白底黑字）
    QString getMessageBoxStyle() const;
};

#endif // WT_PROJECTWIDGET_H
