/*
 * chartwindow.h
 * 文件作用: 通用图表独立窗口头文件
 * 功能描述:
 * 1. 作为一个通用的独立窗口容器，内部包含一个 ChartWidget。
 * 2. 对外提供访问内部 ChartWidget 的接口，以便外部配置数据和模式。
 */

#ifndef CHARTWINDOW_H
#define CHARTWINDOW_H

#include <QWidget>
#include "chartwidget.h"

namespace Ui {
class ChartWindow;
}

class ChartWindow : public QWidget
{
    Q_OBJECT

public:
    // 构造函数
    explicit ChartWindow(QWidget *parent = nullptr);
    // 析构函数
    ~ChartWindow();

    // 获取内部的 ChartWidget 指针
    ChartWidget* getChartWidget();

    // 转发信号：当内部 ChartWidget 请求导出数据时
    // 注意：如果需要在独立窗口中实现“部分导出选点”，需要在 ChartWindow 或其调用者中处理逻辑
    // 这里我们简单暴露 ChartWidget，由调用者统一连接信号
signals:
    // 可选：转发 ChartWidget 的信号
    void exportDataTriggered();

private:
    Ui::ChartWindow *ui;
};

#endif // CHARTWINDOW_H
