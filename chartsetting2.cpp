/*
 * chartsetting2.cpp
 * 文件作用：双坐标系图表设置对话框实现
 * 功能描述：
 * 1. 加载和应用双坐标系（压力-产量）的各项设置
 * 2. [修复] X轴的网格、可见性设置现在同时作用于上下两个坐标轴
 * 3. [修复] X轴刻度值仅控制下方坐标轴
 */

#include "chartsetting2.h"
#include "ui_chartsetting2.h"

ChartSetting2::ChartSetting2(QCustomPlot* plot, QCPAxisRect* top, QCPAxisRect* bottom, QCPTextElement* title, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ChartSetting2),
    m_plot(plot),
    m_topRect(top),
    m_bottomRect(bottom),
    m_title(title)
{
    ui->setupUi(this);
    loadSettings();
}

ChartSetting2::~ChartSetting2()
{
    delete ui;
}

void ChartSetting2::loadSettings()
{
    // --- 通用设置 ---
    if(m_title) ui->lineChartTitle->setText(m_title->text());

    // X轴设置 (主要读取下方坐标轴的状态，因为它是主轴)
    QCPAxis* xAxis = m_bottomRect->axis(QCPAxis::atBottom);
    ui->lineXLabel->setText(xAxis->label());
    ui->spinXMin->setValue(xAxis->range().lower);
    ui->spinXMax->setValue(xAxis->range().upper);

    ui->checkXVisible->setChecked(xAxis->visible());
    ui->checkXTickLabels->setChecked(xAxis->tickLabels());
    ui->checkXGrid->setChecked(xAxis->grid()->visible());
    ui->checkXSubGrid->setChecked(xAxis->grid()->subGridVisible());

    // --- 压力图设置 (Top Rect, Left Axis) ---
    QCPAxis* pAxis = m_topRect->axis(QCPAxis::atLeft);
    ui->linePLabel->setText(pAxis->label());
    ui->spinPMin->setValue(pAxis->range().lower);
    ui->spinPMax->setValue(pAxis->range().upper);

    ui->checkPVisible->setChecked(pAxis->visible());
    ui->checkPTickLabels->setChecked(pAxis->tickLabels());
    ui->checkPGrid->setChecked(pAxis->grid()->visible());
    ui->checkPSubGrid->setChecked(pAxis->grid()->subGridVisible());

    // --- 产量图设置 (Bottom Rect, Left Axis) ---
    QCPAxis* qAxis = m_bottomRect->axis(QCPAxis::atLeft);
    ui->lineQLabel->setText(qAxis->label());
    ui->spinQMin->setValue(qAxis->range().lower);
    ui->spinQMax->setValue(qAxis->range().upper);

    ui->checkQVisible->setChecked(qAxis->visible());
    ui->checkQTickLabels->setChecked(qAxis->tickLabels());
    ui->checkQGrid->setChecked(qAxis->grid()->visible());
    ui->checkQSubGrid->setChecked(qAxis->grid()->subGridVisible());
}

void ChartSetting2::on_buttonBox_accepted()
{
    applySettings();
    accept();
}

void ChartSetting2::applySettings()
{
    // 1. 通用标题
    if(m_title) m_title->setText(ui->lineChartTitle->text());

    // 2. X轴设置 (关键修改：同步应用到上下两个X轴)
    QCPAxis* xAxisBottom = m_bottomRect->axis(QCPAxis::atBottom); // 下方X轴
    QCPAxis* xAxisTop = m_topRect->axis(QCPAxis::atBottom);       // 上方X轴

    // 范围和标签
    xAxisBottom->setLabel(ui->lineXLabel->text());
    xAxisBottom->setRange(ui->spinXMin->value(), ui->spinXMax->value());
    xAxisTop->setRange(ui->spinXMin->value(), ui->spinXMax->value()); // 范围必须同步

    // [修复] 可见性、网格、子网格：上下同时生效
    bool xVisible = ui->checkXVisible->isChecked();
    bool xGrid = ui->checkXGrid->isChecked();
    bool xSubGrid = ui->checkXSubGrid->isChecked();

    xAxisBottom->setVisible(xVisible);
    xAxisTop->setVisible(xVisible);

    xAxisBottom->grid()->setVisible(xGrid);
    xAxisTop->grid()->setVisible(xGrid);

    xAxisBottom->grid()->setSubGridVisible(xSubGrid);
    xAxisTop->grid()->setSubGridVisible(xSubGrid);

    // [修复] 刻度值：仅控制下方，上方强制关闭（避免中间文字重叠）
    xAxisBottom->setTickLabels(ui->checkXTickLabels->isChecked());
    xAxisTop->setTickLabels(false);

    // 3. 压力Y轴设置
    QCPAxis* pAxis = m_topRect->axis(QCPAxis::atLeft);
    pAxis->setLabel(ui->linePLabel->text());
    pAxis->setRange(ui->spinPMin->value(), ui->spinPMax->value());
    pAxis->setVisible(ui->checkPVisible->isChecked());
    pAxis->setTickLabels(ui->checkPTickLabels->isChecked());
    pAxis->grid()->setVisible(ui->checkPGrid->isChecked());
    pAxis->grid()->setSubGridVisible(ui->checkPSubGrid->isChecked());

    // 4. 产量Y轴设置
    QCPAxis* qAxis = m_bottomRect->axis(QCPAxis::atLeft);
    qAxis->setLabel(ui->lineQLabel->text());
    qAxis->setRange(ui->spinQMin->value(), ui->spinQMax->value());
    qAxis->setVisible(ui->checkQVisible->isChecked());
    qAxis->setTickLabels(ui->checkQTickLabels->isChecked());
    qAxis->grid()->setVisible(ui->checkQGrid->isChecked());
    qAxis->grid()->setSubGridVisible(ui->checkQSubGrid->isChecked());

    m_plot->replot();
}
