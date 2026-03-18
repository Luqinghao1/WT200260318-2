/*
 * chartwindow.cpp
 * 文件作用: 通用图表独立窗口实现文件
 * 功能描述:
 * 1. 初始化界面，包含 ChartWidget。
 * 2. 将内部 ChartWidget 的信号转发出来（可选）。
 */

#include "chartwindow.h"
#include "ui_chartwindow.h"

ChartWindow::ChartWindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChartWindow)
{
    ui->setupUi(this);

    // 设置窗口属性，使其独立显示时销毁释放内存
    setAttribute(Qt::WA_DeleteOnClose);

    // 连接内部信号 (如果需要转发)
    connect(ui->chartWidget, &ChartWidget::exportDataTriggered, this, &ChartWindow::exportDataTriggered);
}

ChartWindow::~ChartWindow()
{
    delete ui;
}

ChartWidget* ChartWindow::getChartWidget()
{
    return ui->chartWidget;
}
